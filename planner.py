from transformers import pipeline

def create_weekly_plan(prompt, model=None):
    try:
        if model is None:
            model = pipeline(
                "text-generation",
                model="HooshvareLab/gpt2-fa",
                tokenizer="HooshvareLab/gpt2-fa",
                max_length=512,
                pad_token_id=5
            )

        input_text = f"Weekly program for: {prompt}\n"
        output = model(
            input_text,
            max_new_tokens=200,
            do_sample=True
        )
        
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
            response_text = output[0].get("generated_text", "")
        else:
            response_text = str(output)
        
        response_text = response_text.replace(input_text, "").strip()
        
        if not response_text:
            response_text = "متأسفانه برنامه‌ای تولید نشد. لطفاً دوباره تلاش کنید."
        
        week_num = len(prompt) % 10 

        return response_text, week_num
    except Exception as e:
        return f"خطا در تولید برنامه هفتگی: {str(e)}", "error"