import datetime


def create_weekly_plan(user_input, llm):
    try:
        now = datetime.datetime.now()
        if hasattr(now.isocalendar(), "week"):
            current_week = now.isocalendar().week
        else:
            current_week = now.isocalendar()[1]
        print(f"Current week: {current_week}")

    except Exception as e:
        print(f"Error in calculating the week number: {str(e)}")
        current_week = None

    prompt = f"""Based on the information below, create a suitable weekly schedule:
    student_request: {user_input}
    week: {current_week}

    Please provide a detailed schedule for each day of the week.
    For each lesson, indicate the time for study, rest, and other activities"""

    response = llm(prompt)
    return response, current_week