from datetime import datetime, timedelta
import calendar

def create_weekly_plan(request, llm):
    try:
        if "Weekly program for" in request or "I want a weekly schedule." in request:
            subject = request.split("for")[-1].strip()
            plan = f"Your weekly schedule for{subject} \n"

            week_num = str(datetime.now().isocalendar()[1])
            start_date = datetime.now()
            for i in range(7):
                day = start_date + timedelta(days=i)
                plan += f"{calendar.day_name[day.weekday()]}: {subject} - ساعت 9 صبح تا 12 ظهر\n"
            
            return plan, week_num
        else:
            return "Your request is not clear, please write your request for a weekly schedule more precisely.", None
    except Exception as e:
        return f"Error creating weekly schedule: {str(e)}", None
