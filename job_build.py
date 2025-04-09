import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import build
import re

def crop_with_tags_and_overwrite(file_path,content):

    pattern = r"(\\begin\{document\}.*?\\end\{document\})"
    match = re.search(pattern, content, re.DOTALL)

    if match:
        cropped_with_tags = match.group(1).strip()
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cropped_with_tags)
        print(f"Successfully overwritten {file_path} with document content including tags.")
    else:
        raise ValueError("Could not find \\begin{document} and \\end{document} block.")

class JobBuild:
    def __init__(self, resume, form_data, to_add, job_dis):
        self.resume = resume
        self.form_data = form_data
        self.to_add = to_add
        self.job_dis = job_dis
        EDITOR_PROMPT = '''You are provided with a resume template in LaTeX format that is divided into two parts:

                        1. **Imports Section:** This section contains all necessary LaTeX import commands and configurations. **DO NOT modify any content in this section.**

                        2. **Body Section:** This section contains the actual resume content that can be edited.

                        In addition to the resume template, you are supplied with three distinct data sources:
                        - **form_data:** Contains the current or old resume information, including fields like projects, skills, POR (Positions of Responsibility), achievements, and experience.
                        form_data
                        {form_data}
                        - **to_add:** Contains the new or updated data that should be edited into the resume. These fields follow the same structure as `form_data` (projects, skills, POR, achievements, and experience) but represent changes or additions.
                        to_add
                        {to_add}
                        - **job_dis:** The job description for the role to which the resume should be specifically tailored.
                        job_dis
                        {job_dis}

                        Your task is to update the **Body Section** of the LaTeX resume by merging the content from `form_data` and `to_add`. Do not necessarily include all items from every field; choose and edit only the most relevant content to make the resume highly targeted and specific to the job description provided in `job_dis`.

                        Follow these strict instructions:
                        - **DO NOT modify** anything in the Imports Section.
                        - **Stick to the Template** don't try to add custom commands that might lead to compilation errors.
                        - **DO NOT make** same section multiple times. Merge the content in the section itself.
                        - **Merge Data:** Replace or integrate the corresponding sections in `form_data` with the new information from `to_add` where applicable.
                        - **Tailor Precisely:** Edit the resume content to directly align with the job description (`job_dis`) and the desired role, using industry-standard keywords, measurable results, and action-oriented language.
                        - **Conciseness & Specificity:** Ensure the full resume fits on a single page. Carefully choose and include only the most relevant projects, skills, POR, achievements, and experience.
                        - **Formatting:** Use clean bullet points and concise paragraphs. Maintain clarity and ensure a professional layout.
                        - **Consistency:** Apply uniform formatting, punctuation, and language style throughout the document.
                        - **Professional Tone:** The resume should read as a polished, professional document aimed at the specific job role.
                        
                        **Output:**  
                        Return the complete LaTeX resume as your final output. Begin with the unaltered Imports Section, followed by the modified Body Section that integrates the merged data and is tailored to the provided job description. The output must include only the final LaTeX code without any additional commentary.

                        Here is the input resume template:

                        {resume}
                        '''

        self.initial_template = EDITOR_PROMPT.format(
            form_data=self.form_data,
            to_add = self.to_add,
            job_dis=self.job_dis,
            resume=self.resume
            )
    
        llm = ChatGroq(model= 'deepseek-r1-distill-qwen-32b',  groq_api_key=os.getenv("ARMAN_GROQ_API_KEY"))
        self.prompt = ChatPromptTemplate.from_messages([
        ("system", self.initial_template),
        ("human", "{input}")
        ])
        self.build_chain = self.prompt | llm
    
    def invoke(self, query):
        response = self.build_chain.invoke({
                "input": query
            })
        return response.content
        
        
if __name__ == '__main__':
    with open("user_data.json", "r") as file:
        form_data = str(json.load(file))
        form_data = form_data.replace("{", "{{").replace("}", "}}")
    with open("jobDis.txt", "r") as job:
        jobDis = job.read()
    with open('to_add.json', 'r') as add:
        toAdd = add.read()
        toAdd = toAdd.replace("{", "{{").replace("}", "}}")
    start = True
    while(True):  
        with open("main.tex", "r", encoding="utf-8") as tex_file:
            resume_data = tex_file.read()
        resume = resume_data.replace("{", "{{").replace("}", "}}")
        if start:
            obj_job = JobBuild(resume=resume,form_data=form_data,to_add=toAdd, job_dis=jobDis)
            response = obj_job.invoke(query= "Please tailor my resume according to my job discription")
            start = False
        else:
            obj_edit = build.Build(resume=resume, form_data=form_data)
            query = input("\n\nEnter Query ---->\n\n")
            response = obj_edit.invoke(query= str(query))
        print(response)
        cropped = crop_with_tags_and_overwrite("main.tex",content = str(response))
        print(cropped) 