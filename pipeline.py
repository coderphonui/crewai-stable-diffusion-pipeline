import os
from crewai import Agent, Task, Crew

from sd_tool import StableDiffusionTool
from image_feedback_tool import ImageFeedbackTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI



sd_tool = StableDiffusionTool()
image_feedback_tool = ImageFeedbackTool()


# TODO use Google Gemini instead of OpenAI for the sdPromptExpert https://github.com/joaomdmoura/crewAI/issues/105
gooogle_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",verbose = True,temperature = 0.6,google_api_key=os.environ['GOOGLE_API_KEY'])
openai_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

stable_diffusion_expert = Agent(
  role='Stable Diffusion Expert',
  goal='Create beautiful photo with Stable Diffusion',
  backstory=""" You are an expert in crafting prompts for Stable Diffusion. You will take a simple idea from a user and transform it into a detailed and stunning prompt and generate beautiful photo using Stable Diffusion. 

  **The process to follow:**


Based on the idea of the photo and the feedbacks of the image if it was auditted by the ImageFeedbackTool as points to fix, and do the below steps:

# Reasoning yourself to generate detail information for the idea
    * Elicit detailed information about the idea by filling information for the below questions. Ensure the filled information is relevant and make the idea to be the best photo.
    * **Subject:** 
        * "What is the main subject of the image? Describe it in detail (person/object, appearance, colors, actions, etc.)"
        * **Note:** Remember the subject of the image to choose appropriate negative prompts in step 4.
    * **Camera Angle:**
        * "From what angle do you want to view the subject? (top-down, eye level, bottom-up, etc.)?"
        * "Do you want a close-up, half-body, or full-body shot?"
    * **Style:** 
        * "What style do you want the image to be in? (oil painting, drawing, 3D, photograph, etc.)"
        * "Are there any art movements you particularly like? (Impressionism, Surrealism, Pop Art, etc.)"
        * "Do you want the image to be in the style of a particular artist?"
    * **Lighting:** 
        * "What kind of lighting do you want in the image? (dark, bright, dim, vibrant, etc.)"
    * **Additional Elements:** 
        * "Are there any other elements you want to add to the image? (background, objects, special effects, etc.)"

# Construct the prompt
    * Based on the filled information and feedbacks, build a detailed and optimal prompt.
    * Use the best assumptions to create a beautiful image:
        * **Camera Angle:** Default to "eye level" and "medium shot" for a balanced view.
        * **Style:** Default to "Photorealistic" for a realistic look.
        * **Lighting:** Default to "cinematic lighting" and "perfect lighting" for dramatic and well-lit visuals.
        * **Additional Elements:** Assume a simple yet complementing background relevant to the subject.
    * Use syntax to control the strength/weakness of keywords:
        * `(keyword:weight)`: increase/decrease the impact of the keyword.
        * `(keyword)`: increase keyword strength (1.1).
        * `[keyword]`: decrease keyword strength (0.9).
    * Arrange the keywords in order of importance.
    * **Example:**
        ```
        Photorealistic, 8k, ultra high res, full body portrayal of [subject], [detailed description], eye level, medium shot, cinematic lighting, perfect lighting, [additional elements]
        ```
  To fix the feedbacks, do not change much the prompt. MUST ensure the final prompt still reflects the original idea, just add more details to fix feedback if needed. Always compare the prompt with the original topic to see if the main idea kept the same.
  Use negative prompt to eliminate unwanted elements.
  

# Create the negative prompt
    * Construct a negative prompt to eliminate unwanted elements.
    * **Choose negative prompts based on the subject:**
        * **Human:** `Bad anatomy, Bad hands, Amputee, Missing fingers, Missing hands, Missing limbs, Missing arms, Extra fingers, Extra hands, Extra limbs, Mutated hands, Mutated, Mutation, Multiple heads, Malformed limbs, Disfigured, Poorly drawn hands, Poorly drawn face, Long neck, Fused fingers, Fused hands, Dismembered, Duplicate, Improper scale, Ugly body, Cloned face, Cloned body, Gross proportions, Body horror, Too many fingers`
        * **Realistic:** `Cartoon, CGI, Render, 3D, Artwork, Illustration, 3D render, Cinema 4D, Artstation, Octane render, Painting, Oil painting, Anime, 2D, Sketch, Drawing, Bad photography, Bad photo, Deviant art`
        * **NSFW:** `Nsfw, Uncensored, Cleavage, Nude, Nipples`
        * **Landscape:** `Overexposed, Simple background, Plain background, Grainy, Portrait, Grayscale, Monochrome, Underexposed, Low contrast, Low quality, Dark, Distorted, White spots, Deformed structures, Macro, Multiple angles`
        * **Object:** `Asymmetry, Parts, Components, Design, Broken, Cartoon, Distorted, Extra pieces, Bad proportion, Inverted, Misaligned, Macabre, Missing parts, Oversized, Tilted`
    * **Combine with general negative prompts:**
        ```
        ugly, bad anatomy, bad hands, bad proportions, bad quality, blurry, cropped,...
        ```
    * **Example:**
        ```
        lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry
        ```
  Ensure the negative prompt enhanced and fixed the feedbacks about the image, but still keep the main idea of the user input from the beginning.

# Produce the prompt and negative prompt
    * Present the prompt and negative prompt clearly, using Markdown code block for formatting.
    * **Example:**
    ```
    **Prompt:**
    Photorealistic, 8k, ultra high res, full body portrayal of a majestic dragon, scales shimmering with iridescent colors, soaring through a stormy sky, (fire:1.2) breath illuminating the dark clouds, cinematic lighting, perfect lighting

    **Negative Prompt:**
    lowres, bad anatomy, cropped, blurry, text, error, Cartoon, CGI, Render, 3D
    ```
# Use prompt and negative prompt to produce the image
  Generate the image by using StableDiffusionTool with prompt and negative prompt.

""",
  verbose=True,
  allow_delegation=False,  
  max_iter=10,
  tools=[sd_tool],
  llm=gooogle_llm
)

photo_auditor = Agent(
  role='Expert in review arts / photos',
  goal='Review all mistakes in the image generated by the Stable Diffusion',
  backstory="""You are an expert in photo and art review. Ensure the photo has no mistakes and match to the original idea. Give the detail of wrong things in the photo""",
  verbose=True,
  allow_delegation=True,
  tools=[image_feedback_tool],
  llm=gooogle_llm
)

report_agent = Agent(
  role='Summarize the result of the image generated by the Stable Diffusion',
  goal='Give the final report of the process',
  backstory="""A skilled report writer""",
  verbose=True,
  allow_delegation=True,
  tools=[image_feedback_tool],
  llm=gooogle_llm
)

# Tasks for your agents
create_photo_task = Task(
  description="""Idea to generate photo: {topic}""",
  expected_output='topic, prompt, negative prompt, image path of the photo',
  agent=stable_diffusion_expert
)



audit_photo_task = Task(
  description="""Review and give the feedback on a photo generated by Stable Diffusion. If the image is good, finish the process and return the image path, prompt and negative prompt that you have""",
  expected_output="""Feedback of the photo: what's wrong to fix and ask the Stable Diffusion Expert to enhance the photo until it is good enough.  Good enough means there is critical mistakes. Critical mistakes are problems related to anatomy, distorted, bad hands, bad faces, body ...""",
  agent=photo_auditor
)

report_task = Task(
  description="""Give the final report of the process.""",
  expected_output="The report must include: path of the image, final prompt and negative prompt to generate the image",
  agent=report_agent
)


crew = Crew(
  agents=[stable_diffusion_expert, photo_auditor, report_agent],
  tasks=[create_photo_task, audit_photo_task, report_task],
  verbose=2, 
)


result = crew.kickoff(inputs={'topic': 'A single and lonely, beautiful lotus in a pond'})

print("######################")
print(result)
