from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import uvicorn
import os

app = FastAPI()

# 1. CORS 설정 (Flutter 로컬 및 웹 테스트 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Gemini API 설정 (보안: 환경변수에서만 키 로드)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
genai.configure(api_key=api_key)

# 3. Vision 지원 모델
model = genai.GenerativeModel("gemini-2.5-flash")


def _parse_quiz_json(result_text: str, max_count: int) -> list:
    """Gemini 응답 텍스트에서 JSON 배열 파싱. 공통 로직."""
    if result_text.startswith("```json"):
        result_text = result_text[7:]
    if result_text.startswith("```"):
        result_text = result_text[3:]
    if result_text.endswith("```"):
        result_text = result_text[:-3]
    result_text = result_text.strip()
    quiz_data = json.loads(result_text)
    if not isinstance(quiz_data, list):
        quiz_data = [quiz_data] if isinstance(quiz_data, dict) else []
    return quiz_data[:max_count]


@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    start_page: int = Form(...),
    end_page: int = Form(...),
    question_count: int = Form(...),
    question_type: str = Form(...),
    language: str = Form(...),
    custom_prompt: str = Form(None),
):
    print(f"=== [DEBUG] 들어온 커스텀 프롬프트: {custom_prompt} ===")
    pdf_document = None
    try:
        # 4. PDF 파일 읽기
        contents = await file.read()
        pdf_document = fitz.open(stream=contents, filetype="pdf")

        num_pages = len(pdf_document)
        if start_page < 1 or end_page > num_pages or start_page > end_page:
            raise HTTPException(
                status_code=400,
                detail="유효하지 않은 페이지 범위입니다.",
            )

        total_pages = end_page - start_page + 1
        if total_pages > 50:
            raise HTTPException(
                status_code=400,
                detail="한 번에 최대 50페이지까지 선택할 수 있습니다.",
            )
        if question_count < 1 or question_count > 100:
            raise HTTPException(
                status_code=400,
                detail="문제 수는 1~100개 사이로 지정해 주세요.",
            )

        # 5. PDF 페이지를 고화질 이미지(PIL)로 변환
        images_to_process = []
        for page_num in range(start_page - 1, end_page):
            page = pdf_document.load_page(page_num)
            zoom = 2.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images_to_process.append(img)

        if not images_to_process:
            raise HTTPException(
                status_code=400,
                detail="변환할 수 있는 PDF 페이지가 없습니다.",
            )

        # 6. 다듬은 프롬프트 (역할·제약·출력 형식 명확화)
        custom_prompt_block = f"""
[사용자 특별 지시사항 (절대 규칙 - 최우선 반영)]
아래 내용은 사용자가 직접 작성한 커스텀 요청사항입니다. 기본 규칙과 충돌하더라도 아래 내용을 무조건 최우선으로 반영하여 문제를 출제하세요. (예: 힌트 추가, 특정 말투 사용 등)
"{custom_prompt}"
""" if custom_prompt else ""

        # question_type에 따라 해당 유형 규칙만 포함
        if question_type == "4지선다":
            type_rule_block = """
3. '4지선다' 유형별 규칙:
   - 4지선다: 
     * options는 반드시 4개.
     * 유일무이한 정답: 4개의 선지 중 정답은 오직 단 1개여야 하며, 논란의 여지가 없어야 합니다.
     * 상호 배타적 선지: 선지들끼리 의미가 겹치거나 한 선지가 다른 선지를 포괄하지 않도록 완전히 독립적으로 구성하세요.
     * 매력적이지만 명백한 오답: 나머지 3개의 오답 선지는 사용자가 헷갈릴 만큼 주제와 관련성이 높아야 하지만, 객관적 사실(PDF 내용)에 명백하게 위배되는 '100% 오답'으로만 구성하세요.
"""
        elif question_type == "OX":
            type_rule_block = """
3. 'OX' 유형별 규칙:
   - OX: options는 반드시 ["O", "X"].
"""
        elif question_type == "단답식":
            type_rule_block = """
3. '단답식' 유형별 규칙:
   - 단답식: 
     * options는 반드시 빈 배열 [].
     * 정답(answer)은 무조건 15글자 이내의 핵심 명사, 고유명사, 혹은 숫자로만 작성하세요.
     * 정답에 띄어쓰기, 특수기호, 쉼표(,), '및/과/와' 같은 연결어나 조사가 절대 포함되지 않게 하세요. (문장형, 서술형 절대 금지)
     * 문제는 매우 명확하고 구체적으로 출제하여, 다른 해석의 여지 없이 오직 하나의 정답만 도출되도록 만드세요.
"""
        else:
            # 예외적으로 알 수 없는 유형이 들어오면 기존 전체 규칙을 그대로 사용
            type_rule_block = """
3. '{question_type}' 유형별 규칙:
   - 4지선다: 
     * options는 반드시 4개.
     * 유일무이한 정답: 4개의 선지 중 정답은 오직 단 1개여야 하며, 논란의 여지가 없어야 합니다.
     * 상호 배타적 선지: 선지들끼리 의미가 겹치거나 한 선지가 다른 선지를 포괄하지 않도록 완전히 독립적으로 구성하세요.
     * 매력적이지만 명백한 오답: 나머지 3개의 오답 선지는 사용자가 헷갈릴 만큼 주제와 관련성이 높아야 하지만, 객관적 사실(PDF 내용)에 명백하게 위배되는 '100% 오답'으로만 구성하세요.
   - OX: options는 반드시 ["O", "X"].
   - 단답식: 
     * options는 반드시 빈 배열 [].
     * 정답(answer)은 무조건 15글자 이내의 핵심 명사, 고유명사, 혹은 숫자로만 작성하세요.
     * 정답에 띄어쓰기, 특수기호, 쉼표(,), '및/과/와' 같은 연결어나 조사가 절대 포함되지 않게 하세요. (문장형, 서술형 절대 금지)
     * 문제는 매우 명확하고 구체적으로 출제하여, 다른 해석의 여지 없이 오직 하나의 정답만 도출되도록 만드세요.
"""

        prompt = f"""
당신은 교육용 퀴즈 출제 전문가입니다. 아래에 제공된 이미지들은 **한 PDF 문서의 페이지를 순서대로** 촬영한 것입니다. 이미지를 시각적으로 분석하여 텍스트·수식·표·도표를 정확히 읽어내세요.

[작업 지시]
1. 이미지에서 읽은 내용을 바탕으로, 문서의 **주 사용 언어**를 판단하세요.
2. 판단한 언어와 **동일한 언어**로 정확히 {question_count}개의 '{question_type}' 형식 퀴즈를 생성하세요. (예: 문서가 영어면 문제·보기·해설 모두 영어, 한국어면 모두 한국어)
3. 문제는 반드시 제공된 이미지(PDF 내용)에 기반해야 하며, 이미지에 없는 내용을 지어내지 마세요.

[엄격한 제약]
1. 보기(options) 각 항목: 20자 이하. 핵심만 간결하게.
2. 해설(explanation): 40자 이하. 간단명료하게.
{type_rule_block}
4. 출력은 **오직 유효한 JSON 배열 하나**만 하세요. ```json 같은 마크다운, 설명, 주석, 접두/접미 문구는 절대 포함하지 마세요.
{custom_prompt_block}
[출력 형식]
[
  {{"question": "문제 문장", "options": ["A", "B", "C", "D"], "answer": "정답", "explanation": "해설"}},
  ...
]
"""

        # 7. Gemini API 호출 (프롬프트 + 이미지 리스트, SDK가 PIL을 Blob으로 변환)
        response = model.generate_content([prompt] + images_to_process)

        # 8. JSON 응답 파싱 및 마크다운 클리닝
        result_text = response.text.strip()
        quiz_data = _parse_quiz_json(result_text, question_count)

        return {"quizzes": quiz_data}

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"퀴즈 JSON 파싱 실패: {str(e)}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서버 에러 발생: {str(e)}",
        )
    finally:
        if pdf_document is not None:
            pdf_document.close()


@app.post("/pdf-page-count")
async def pdf_page_count(file: UploadFile = File(...)):
    """업로드된 PDF의 총 페이지 수 반환."""
    pdf_document = None
    try:
        contents = await file.read()
        pdf_document = fitz.open(stream=contents, filetype="pdf")
        num_pages = len(pdf_document)
        if num_pages <= 0:
            raise HTTPException(status_code=400, detail="PDF 페이지 수를 확인할 수 없습니다.")
        return {"total_pages": num_pages}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 페이지 수 계산 실패: {str(e)}")
    finally:
        if pdf_document is not None:
            pdf_document.close()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
