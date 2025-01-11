from langchain_core.messages import HumanMessage, AIMessage
from fastapi import APIRouter, HTTPException
from app.services import (
    MessagesState,
    generate_question_with_llm,
    score_and_provide_feedback,
)
from app.config import llm
from app.models import UserInput, UserResponse

router = APIRouter()
conversation_state = MessagesState()


@router.post("/start")
async def start_interview(user_input: UserInput):
    try:
        # Reset conversation state for a new interview
        conversation_state.messages.clear()
        conversation_state.user_info = {
            "candidate_id": user_input.candidate_id,  # Store candidateId
            "candidate_name": user_input.candidate_name.strip(),
            "email": user_input.email.strip(),
            "job_title": user_input.job_title.strip(),
            "experience": user_input.experience.strip(),
        }
        conversation_state.user_answers.clear()
        conversation_state.correct_answers.clear()
        conversation_state.asked_questions.clear()

        # Validate candidateId and job title
        if not conversation_state.user_info.get("candidate_id"):
            raise HTTPException(status_code=400, detail="Candidate ID is missing.")
        if not conversation_state.user_info.get("job_title"):
            raise HTTPException(status_code=400, detail="Job title is missing in user input.")

        # Generate the first question
        question = generate_next_question(
            llm, conversation_state, user_input.job_title, user_input.experience
        )
        conversation_state.messages.append(AIMessage(content=question))

        return {"message": "Interview started", "question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting interview: {str(e)}")


@router.post("/next_question")
async def next_question(user_response: UserResponse):
    try:
        # Store the user's response
        conversation_state.messages.append(HumanMessage(content=user_response.user_response.strip().lower()))
        conversation_state.user_answers.append(user_response.user_response.strip().lower())

        # Retrieve user information
        job_title = conversation_state.user_info.get("job_title", "").strip()
        experience = conversation_state.user_info.get("experience", "").strip()

        # Validate candidateId and job title
        if not conversation_state.user_info.get("candidate_id"):
            raise HTTPException(status_code=400, detail="Candidate ID is missing.")
        if not job_title:
            raise HTTPException(status_code=400, detail="Job title is missing.")

        # Check if the interview is complete
        if len(conversation_state.user_answers) >= conversation_state.total_questions:
            feedback_details = score_and_provide_feedback(conversation_state)
            return {"complete": True, "feedback": feedback_details}

        # Generate the next question
        question = generate_next_question(llm, conversation_state, job_title, experience)
        conversation_state.messages.append(AIMessage(content=question))
        return {"message": "Next question generated", "question": question}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching next question: {str(e)}")


@router.post("/get_feedback")
async def get_feedback():
    try:
        # Ensure enough questions have been answered
        if len(conversation_state.user_answers) < conversation_state.total_questions:
            raise HTTPException(
                status_code=400, detail="Not enough answers to provide feedback."
            )

        # Generate feedback
        feedback_details = score_and_provide_feedback(conversation_state)

        # Log the candidate feedback (optional: save to database here if required)
        print(f"Candidate Feedback for {conversation_state.user_info.get('candidate_id')}: {feedback_details}")

        return {"feedback": feedback_details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")


def generate_next_question(
    llm, state: MessagesState, job_title: str, experience: str
) -> str:
    if not job_title:
        raise ValueError("Job title is missing. Ensure the user info includes a valid job title.")

    # Always generate multiple-choice questions
    question_type = "multiple-choice"

    # Generate a unique multiple-choice question
    question_data = generate_question_with_llm(
        llm, job_title, experience, question_type, state
    )
    return question_data
