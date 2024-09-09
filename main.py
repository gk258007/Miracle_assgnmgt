from typing import List
import logging
import sys
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from PIL import Image
import pytesseract
import io


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Model loaded successfully. Using device: {device}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    sys.exit(1)

class TestCase(BaseModel):
    description: str
    preconditions: str
    steps: List[str]
    expected_result: str

async def process_image(image_file: UploadFile) -> Image.Image:
    try:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents))
        return image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

def extract_text_from_image(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image)
        logger.info(f"Extracted text from image: {text}")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return ""

async def generate_testing_instructions(context: str, images: List[Image.Image]) -> List[TestCase]:
    try:
        logger.info(f"Context: {context}")
        logger.info(f"Number of images: {len(images)}")


        extracted_text = ""
        for image in images:
            extracted_text += extract_text_from_image(image)


        prompt = f"""
Given the extracted text and visual elements from the uploaded UI screenshot(s), generate comprehensive test cases for the user interface. For each test case, consider the following:

1. Test Case ID: Assign a unique identifier for each test case.
2. Test Case Description: Provide a concise description of what the test case is intended to verify.
3. Preconditions: List any conditions that must be met before executing the test case. This could include user roles, data states, or navigation steps.
4. Test Steps: Outline the specific steps to perform the test case. Be detailed about each action the user should take.
5. Expected Result: Describe what should happen after completing the test steps. Include what should be displayed or how the system should respond.
6. Postconditions: Specify any state changes or effects that should result from executing the test case.

Use the following context to inform your test case generation: {context}
And the following extracted text from images: {extracted_text}

### Example Output Format

- Test Case ID: TC001
- Test Case Description: Verify the functionality of the 'Search' button on the homepage.
- Preconditions: User is on the homepage and is logged in.
- Test Steps:
  1. Click on the 'Search' button.
  2. Enter a query in the search field.
  3. Press Enter or click on the search icon.
- Expected Result: The search results page should display relevant results based on the query entered.
- Postconditions: User should see search results or an appropriate message if no results are found.

Please ensure that the generated test cases are comprehensive and cover various aspects of the user interface as depicted in the screenshots and provided context.
"""

        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        logger.info(f"Inputs: {inputs}")

        
        output = model.generate(**inputs, max_length=1024)
        logger.info(f"Raw output: {output}")

        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")

        
        test_cases = parse_generated_text_to_test_cases(generated_text)
        logger.info(f"Parsed test cases: {test_cases}")
        return test_cases

    except Exception as e:
        logger.error(f"Error generating testing instructions: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating testing instructions")

def parse_generated_text_to_test_cases(generated_text: str) -> List[TestCase]:
    test_cases = []
    test_case_blocks = generated_text.split("Description:")
    for block in test_case_blocks:
        if block.strip():
            try:
                description = block.split("Preconditions:")[0].strip()
                preconditions = extract_between(block, "Preconditions:", "Testing Steps:")
                steps = extract_between(block, "Testing Steps:", "Expected Result:")
                expected_result = block.split("Expected Result:")[-1].strip()

                test_cases.append(
                    TestCase(
                        description=description,
                        preconditions=preconditions,
                        steps=[step.strip() for step in steps.split("\n") if step.strip()],
                        expected_result=expected_result
                    )
                )
            except Exception as e:
                logger.error(f"Error parsing test case block: {str(e)}")
    
    return test_cases

def extract_between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start)[-1].split(end)[0].strip()
    except IndexError:
        return ""

@app.post("/generate_testing_instructions")
async def generate_instructions(
    context: str = Form(""),
    screenshots: List[UploadFile] = File(...)
):
    try:
        if not screenshots:
            raise HTTPException(status_code=400, detail="At least one screenshot is required")

        logger.info(f"Received request with {len(screenshots)} screenshots and context: {context}")

        images = [await process_image(screenshot) for screenshot in screenshots]
        test_cases = await generate_testing_instructions(context, images)

        return {"testing_instructions": [tc.dict() for tc in test_cases]}
    except HTTPException as he:
        logger.error(f"HTTP exception: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in generate_instructions: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
