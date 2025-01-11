from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, TypedDict, List, Set
from app.config import llm


class MessagesState:
    def __init__(self):
        self.messages: List[AIMessage | HumanMessage] = []
        self.user_info: dict[str, str] = {}  # Include candidateId in user_info
        self.candidate_id: Optional[str] = None  # Explicitly store candidateId
        self.score: int = 0
        self.current_question: int = 0
        self.total_questions: int = 10
        self.user_answers: List[str] = []
        self.correct_answers: List[str] = []
        self.asked_questions: Set[str] = set()  # Set for unique questions


class FeedbackItem(TypedDict):
    question: str
    user_answer: str
    correct_answer: str
    is_correct: bool


class FeedbackSummary(TypedDict):
    correct_count: int
    total_questions: int
    details: List[FeedbackItem]
    final_feedback: str


def generate_question_with_llm(
    llm,
    job_title: str,
    experience: str,
    question_type: str,
    state: MessagesState
) -> str:
    prompt_modifier = (
        f"The questions must be strictly related to the job title '{job_title}' and the candidate's experience level ({experience}). "
        "Avoid generic, unrelated, or off-topic questions. Tailor the content to the technologies and challenges relevant to the job title."
    )

    if question_type == "multiple-choice":
        prompt = (
            f"Generate a unique multiple-choice question for a {experience} professional applying for the position of '{job_title}'. "
            f"Include 4 plausible options, and ensure the question is highly relevant to the job title. "
            f"Only provide the question and options without revealing the answer or explanation. "
            f"{prompt_modifier}"
        )
    elif question_type == "theoretical":
        prompt = (
            f"Generate a unique theoretical question for a {experience} professional applying for the position of '{job_title}'. "
            f"Ensure the question explores advanced or critical aspects of the role. Do not reveal the answer or explanation. "
            f"{prompt_modifier}"
        )
    else:
        raise ValueError(f"Unsupported question type: {question_type}")

    for _ in range(5):
        response = llm.invoke([HumanMessage(content=prompt)])
        question_content = response.content.strip()

        if question_content not in state.asked_questions:
            state.asked_questions.add(question_content)
            return question_content

    raise ValueError("Unable to generate a unique question after multiple attempts.")


def score_and_provide_feedback(state: MessagesState) -> FeedbackSummary:
    correct_answers = 0
    total_questions = len(state.user_answers)

    feedback_summary: FeedbackSummary = {
        "correct_count": 0,
        "total_questions": total_questions,
        "details": [],
        "final_feedback": "",
    }

    asked_questions_list = list(state.asked_questions)

    for i in range(total_questions):
        question = asked_questions_list[i] if i < len(asked_questions_list) else ""
        user_answer = state.user_answers[i]

        # Use LLM to validate the answer
        validation_prompt = (
            f"Question: {question}\n"
            f"User Answer: {user_answer}\n"
            "Is the user's answer correct or incorrect? Provide a brief explanation and the correct answer."
        )
        response = llm.invoke([HumanMessage(content=validation_prompt)])

        # Handle the LLM response properly
        feedback = extract_feedback_from_response(response)

        # Parse the LLM response to determine correctness
        is_correct = "correct" in feedback.lower() and "incorrect" not in feedback.lower()
        correct_answer = extract_correct_answer(feedback)

        # Prepare feedback item
        feedback_item: FeedbackItem = {
            "question": question,
            "user_answer": user_answer,
            "correct_answer": correct_answer if correct_answer else "Not provided",
            "is_correct": is_correct,
        }

        feedback_summary["details"].append(feedback_item)

        if is_correct:
            correct_answers += 1

    # Update feedback summary
    feedback_summary["correct_count"] = correct_answers
    feedback_summary["final_feedback"] = (
        f"Congratulations! You answered {correct_answers} out of {total_questions} correctly. "
        f"Keep up the good work!" if correct_answers >= 6 else
        f"You answered {correct_answers} out of {total_questions} correctly. "
        f"Review the questions and improve your knowledge!"
    )

    return feedback_summary


def extract_feedback_from_response(response) -> str:
    """Extract feedback content from the LLM response."""
    if isinstance(response, list) and len(response) > 0:
        return response[0]["content"].strip() if "content" in response[0] else response[0].strip()
    elif isinstance(response, dict) and "content" in response:
        return response["content"].strip()
    elif isinstance(response, str):
        return response.strip()
    else:
        return "Unable to determine correctness. Please try again."


def extract_correct_answer(feedback: str) -> Optional[str]:
    """Extract the correct answer from the feedback content if available."""
    if "correct answer:" in feedback.lower():
        return feedback.split("Correct answer:")[-1].strip()
    return None
