FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . .
ENV WANDB_API_KEY=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
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]