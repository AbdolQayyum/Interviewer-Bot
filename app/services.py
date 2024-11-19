from langchain_core.messages import HumanMessage, AIMessage
from typing import Optional, TypedDict, List, Set
from app.config import llm
import time

class MessagesState:
    def __init__(self):
        self.messages: List[AIMessage | HumanMessage] = []
        self.user_info: dict[str, str] = {}
        self.score: int = 0
        self.current_question: int = 0
        self.total_questions: int = 20
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

def collect_user_info(
    state: MessagesState,
    user_name: str,
    field: str,
    experience: str,
    years_of_experience: Optional[str] = None
) -> MessagesState:
    state.messages.append(AIMessage(content="What is your name?"))
    state.messages.append(HumanMessage(content=user_name))
    state.user_info['name'] = user_name

    state.messages.append(AIMessage(content=f"Thanks, {user_name}! What’s your field of specialization?"))
    state.messages.append(HumanMessage(content=field))
    state.user_info['field'] = field

    state.messages.append(AIMessage(content="Are you a fresher or experienced professional?"))
    state.messages.append(HumanMessage(content=experience))
    state.user_info['experience'] = experience

    if experience.lower() == "experienced" and years_of_experience is not None:
        state.messages.append(AIMessage(content="How many years of experience do you have?"))
        state.messages.append(HumanMessage(content=years_of_experience))
        state.user_info['years_of_experience'] = years_of_experience
    elif experience.lower() == "experienced" and years_of_experience is None:
        state.messages.append(AIMessage(content="How many years of experience do you have?"))
        state.messages.append(HumanMessage(content="Not provided"))

    return state

def generate_question_with_llm(
    llm,
    field: str,
    experience: str,
    question_type: str,
    state: MessagesState
) -> str:
    prompt_modifier = "Ensure that this question is different from any previous questions and covers a unique aspect of the topic."

    if question_type == "multiple-choice":
        prompt = (
            f"Generate a unique multiple-choice question for an {experience} {field} professional. "
            f"Include 4 options. Only provide the question and options without the answer or explanation. {prompt_modifier}"
        )
    elif question_type == "theoretical":
        prompt = (
            f"Generate a unique theoretical question for an {experience} {field} professional. "
            f"Only provide the question without any answer or explanation. {prompt_modifier}"
        )
    else:
        raise ValueError(f"Invalid question_type: {question_type}. Expected 'multiple-choice' or 'theoretical'.")

    response = llm.invoke([HumanMessage(content=prompt)])
    question_content = response.content.strip()

    while question_content in state.asked_questions:
        response = llm.invoke([HumanMessage(content=prompt)])
        question_content = response.content.strip()

    state.asked_questions.add(question_content)
    return question_content

def ask_questions_one_by_one(
    llm,
    state: MessagesState,
    field: str,
    experience: str
) -> MessagesState:
    for i in range(10):
        question_data = generate_question_with_llm(llm, field, experience, "multiple-choice", state)
        state.messages.append(AIMessage(content=f"Multiple Choice {i+1}: {question_data}"))
        print(f"Multiple Choice {i+1}: {question_data}")
        user_answer = input(f"Your answer for question {i+1}: ")
        state.messages.append(HumanMessage(content=user_answer))
        state.user_answers.append(user_answer)
        state.correct_answers.append("d")  # Simulated correct answer for demonstration
        time.sleep(2)

    for i in range(10):
        question_data = generate_question_with_llm(llm, field, experience, "theoretical", state)
        state.messages.append(AIMessage(content=f"Theoretical Question {i+1}: {question_data}"))
        print(f"Theoretical Question {i+1}: {question_data}")
        user_answer = input(f"Your response for question {i+1}: ")
        state.messages.append(HumanMessage(content=user_answer))
        state.user_answers.append(user_answer)
        state.correct_answers.append("response")  # Simulated correct response for demonstration
        time.sleep(2)

    return state

def collect_answers_and_score(state: MessagesState) -> MessagesState:
    correct_answers = 0
    total_questions = len(state.user_answers)

    for i in range(total_questions):
        if state.user_answers[i] == state.correct_answers[i]:
            correct_answers += 1

    print("\nReview of your answers:")
    for i in range(total_questions):
        correctness = "Correct" if state.user_answers[i] == state.correct_answers[i] else "Wrong"
        print(f"Question {i+1}: Your answer: {state.user_answers[i]} - {correctness}")

    if correct_answers >= 16:
        state.messages.append(AIMessage(content=f"Congratulations! You answered {correct_answers} out of {total_questions} correctly. You're invited for a further interview."))
    else:
        state.messages.append(AIMessage(content=f"You answered {correct_answers} out of {total_questions}. Keep practicing!"))

    return state

def score_and_provide_feedback(state: MessagesState) -> FeedbackSummary:
    correct_answers = 0
    total_questions = len(state.user_answers)

    feedback_summary: FeedbackSummary = {
        "correct_count": 0,
        "total_questions": total_questions,
        "details": [],
        "final_feedback": "",
    }

    # Convert the set to a list to make it indexable
    asked_questions_list = list(state.asked_questions)

    for i in range(total_questions):
        question = asked_questions_list[i] if i < len(asked_questions_list) else ""
        user_answer = state.user_answers[i]
        correct_answer = state.correct_answers[i] if i < len(state.correct_answers) else ""
        is_correct = user_answer == correct_answer

        feedback_item: FeedbackItem = {
            "question": question,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        }

        feedback_summary["details"].append(feedback_item)
        if is_correct:
            correct_answers += 1

    feedback_summary["correct_count"] = correct_answers
    feedback_summary["final_feedback"] = (
        f"Congratulations! You answered {correct_answers} out of {total_questions} correctly. You're invited for a further interview."
        if correct_answers >= 16 else
        f"You answered {correct_answers} out of {total_questions}. Keep practicing!"
    )

    return feedback_summary
