from transformers import pipeline

def create_weekly_plan(prompt, model=None):
    try:
        if model is None:
            # تنظیم pad_token_id و کاهش max_length
            model = pipeline(
                "text-generation",
                model="HooshvareLab/gpt2-fa",
                tokenizer="HooshvareLab/gpt2-fa",
                max_length=512,  # کاهش برای جلوگیری از خطا
                pad_token_id=5  # تنظیم صریح pad_token_id
            )

        input_text = f"Weekly program for: {prompt}\n"
        output = model(
            input_text,
            max_new_tokens=200,  # تنظیم برای جلوگیری از فراتر رفتن از حد
            do_sample=True,
            return_attention_mask=True  # فعال کردن attention_mask
        )
        
        # Ensure output is a string
        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
            response_text = output[0].get("generated_text", "")
        else:
            response_text = str(output)
        
        # Remove the input_text from the response and strip
        response_text = response_text.replace(input_text, "").strip()
        
        # Check for empty response
        if not response_text:
            response_text = "متأسفانه برنامه‌ای تولید نشد. لطفاً دوباره تلاش کنید."
        
        week_num = len(prompt) % 10 

        return response_text, week_num
    except Exception as e:
        return f"خطا در تولید برنامه هفتگی: {str(e)}", "error"