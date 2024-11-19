from fastapi import APIRouter, HTTPException
from app.services import (
    generate_question_with_llm,
    MessagesState,
    score_and_provide_feedback  
)
from app.config import llm
from app.models import UserInput, UserResponse

router = APIRouter()
conversation_state = MessagesState()  # Global state to store conversation progress

@router.post("/start")
async def start_interview(user_input: UserInput):
    try:
        conversation_state.messages.clear()
        conversation_state.user_info = {
            "user_name": user_input.user_name,
            "field": user_input.field,
            "experience": user_input.experience,
            "years_of_experience": user_input.years_of_experience or "",
        }
        conversation_state.user_answers.clear()
        conversation_state.correct_answers.clear()

        question = generate_next_question(
            llm, conversation_state, user_input.field, user_input.experience
        )
        conversation_state.messages.append({"role": "AI", "content": question})

        return {"message": "Interview started", "question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/next_question")
async def next_question(user_response: UserResponse):
    try:
        conversation_state.messages.append(
            {"role": "User", "content": user_response.user_response}
        )
        conversation_state.user_answers.append(user_response.user_response)

        field = conversation_state.user_info.get("field", "")
        experience = conversation_state.user_info.get("experience", "")
        question = generate_next_question(llm, conversation_state, field, experience)

        if len([msg for msg in conversation_state.messages if msg["role"] == "AI"]) >= 20:
            return {"complete": True}

        conversation_state.messages.append({"role": "AI", "content": question})

        return {"message": "Next question generated", "question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get_feedback")
async def get_feedback():
    try:
        if len(conversation_state.user_answers) == 20:
            raise HTTPException(
                status_code=400, detail="Not enough answers to provide feedback."
            )

        feedback_details = score_and_provide_feedback(conversation_state)
        return {"feedback": feedback_details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_next_question(
    llm, state: MessagesState, field: str, experience: str
) -> str:
    question_count = len([msg for msg in state.messages if msg["role"] == "AI"])

    question_type = "multiple-choice" if question_count < 10 else "theoretical"

    question_data = generate_question_with_llm(
        llm, field, experience, question_type, state
    )
    return question_data
