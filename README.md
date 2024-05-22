# What is this project?

This is my experiment to check the CrewAI and do some integration with Stable Diffusion / OpenAI gpt-4o / Google Gemini to create an image generator from a simple idea.

The problem I want to solve:

* When creating a photo with Stable Diffusion, I usually get stuck with the detail idea of the prompt to make the prompt more detail to make the photo better. 
* Even with a good prompt, the generated photo is not good enough. A lot of problems of artifacts, atonomy, distorted ... It requires a loop to enhance the prompt, finetuing the parameters and run again.

I am not a designer guy. So, I expect the photo needs to be good enough and no need to enhance with other tools (dream) :). 
Instead of running this process again and again, I decide to create a pipeline to automate this.

# The idea of the pipeline

![diagram](stable-difffusion-pipeline.png)

The pipeline is based on CrewAI (https://docs.crewai.com/). The concept is very simple. The pipeline has 2 agents: 
* StableDiffusion expert to create prompt and use StableDiffusion UI tool to generate the photo. I am using automatic1111 (https://github.com/AUTOMATIC1111/stable-diffusion-webui) and its API to generate the photo. The integration with Stable Diffusion is in the sd_tool.py
* Photo Auditor: check the photo to identify problems. If there's no problem, it will stop the process. The PhotoAuditor uses the ImageFeedbackTool which will submit the photo to OpenAI gpt-4o to ask for the verification (image_feedback_tool.py).

# How to run the pipeline?

Step 1: setup your environment variables included Google API key and OpenAI API key

```
export GOOGLE_API_KEY=<your-google-api-key>
export OPENAI_API_KEY=<your-google-api-key>
```

Step 2: check the code of the pipeline.py. You can modify the idea of the photo in the create_photo_task

```
create_photo_task = Task(
  description="""Generate photo for the idea: 'A beautiful girl in a jungle'""",
  expected_output='Prompt and negative prompt in the markdown format in code block and the path of the generated image',
  agent=stable_diffusion_expert
)
```

Step 3: run the AUTOMATIC1111 tool

Checkout the AUTOMATIC1111 and set it up. You should check their guideline. It's out of scope of this repository. Ensure you download some great checkpoints(ChilloutMix, DreamSharper ...). Checkpoints are very important to have nice photos :)

To run the AUTOMATIC1111 with API enable, run the below command:

```
./webui.sh --listen --api
```

Step 4: run the pipeline

```
# Install requirements first
pip install -r requirements.txt

# Run the pipeline
python pipeline.py
```

The generated photo is in the output/txt2img directory. I commit some photos in the output/txt2img directory for reference of the output that I made.

Enjoy!