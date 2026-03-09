from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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


class FlashPromptRequest(BaseModel):
    prompt: str
    question_count: int


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
        if question_count < 1 or question_count > 50:
            raise HTTPException(
                status_code=400,
                detail="문제 수는 1~50개 사이로 지정해 주세요.",
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

        prompt = f"""
당신은 교육용 퀴즈 출제 전문가입니다. 아래에 제공된 이미지들은 **한 PDF 문서의 페이지를 순서대로** 촬영한 것입니다. 이미지를 시각적으로 분석하여 텍스트·수식·표·도표를 정확히 읽어내세요.

[작업 지시]
1. 이미지에서 읽은 내용을 바탕으로, 문서의 **주 사용 언어**를 판단하세요.
2. 판단한 언어와 **동일한 언어**로 정확히 {question_count}개의 '{question_type}' 형식 퀴즈를 생성하세요. (예: 문서가 영어면 문제·보기·해설 모두 영어, 한국어면 모두 한국어)
3. 문제는 반드시 제공된 이미지(PDF 내용)에 기반해야 하며, 이미지에 없는 내용을 지어내지 마세요.

[엄격한 제약]
1. 보기(options) 각 항목: 20자 이하. 핵심만 간결하게.
2. 해설(explanation): 40자 이하. 간단명료하게.
3. '{question_type}' 유형별 규칙:
   - 4지선다: options는 반드시 4개.
   - OX: options는 반드시 ["O", "X"].
   - 단답식: options는 반드시 빈 배열 [].
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


@app.post("/flash-prompt")
async def flash_prompt(body: FlashPromptRequest):
    """사용자 프롬프트를 Gemini Flash에 넘겨 퀴즈 question_count개 생성. PDF 없이 텍스트만 사용."""
    prompt_text = (body.prompt or "").strip()
    question_count = body.question_count

    if not prompt_text:
        raise HTTPException(status_code=400, detail="prompt가 비어 있습니다.")
    if question_count < 1 or question_count > 50:
        raise HTTPException(status_code=400, detail="문제 수는 1~50개 사이로 지정해 주세요.")

    prompt = f"""
당신은 교육용 퀴즈 출제 전문가입니다. 아래 [사용자 지시]에 따라 퀴즈를 생성하세요.

[사용자 지시]
{prompt_text}

[작업]
1. 위 지시에 맞는 퀴즈를 정확히 {question_count}개 생성하세요.
2. 사용자가 형식(4지선다, OX, 단답식 등)을 지정했으면 그에 따르고, 없으면 4지선다로 생성하세요.
3. 보기(options) 각 항목: 20자 이하. 해설(explanation): 40자 이하.
4. 4지선다일 때 options는 반드시 4개, OX일 때 ["O", "X"], 단답식일 때 [].
5. 출력은 **오직 유효한 JSON 배열 하나**만 하세요. ```json 같은 마크다운, 설명, 주석은 포함하지 마세요.

[출력 형식]
[
  {{"question": "문제 문장", "options": ["A", "B", "C", "D"], "answer": "정답", "explanation": "해설"}},
  ...
]
"""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        quiz_data = _parse_quiz_json(result_text, question_count)
        return {"quizzes": quiz_data}
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"퀴즈 JSON 파싱 실패: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서버 에러 발생: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
