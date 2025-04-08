import json
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
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

class Build:
    def __init__(self, resume, form_data):
        self.resume = resume
        self.form_data = form_data
        
        EDITOR_PROMPT = '''You are provided with a resume template in LaTeX format that is divided into two parts:

                            1. **Imports Section:** This section contains all necessary LaTeX import commands and configurations. **DO NOT modify any content in this section.**

                            2. **Body Section:** This section contains the actual resume content that can be edited.

                            Your task is to update the **Body Section** of the resume based on the user's provided details. The user has supplied the following:
                            - **Desired Role:** {desired_role}
                            - **Name:** {name}
                            - **Entry Level:** {role_level}

                            When updating the resume, please ensure the following:

                            - **Keep it Concise:** The resume should be formatted to fit on one page.
                            - **Strong Keywords & Quantifiers:** Use industry-standard keywords, measurable quantifiers, and action-oriented language to showcase achievements and skills.
                            - **Formatting:** Improve the structure and formatting for clarity and visual appeal. Use bullet points or concise paragraphs where appropriate.
                            - **Consistency:** Preserve the imports from the first section without making any changes.
                            - **Professional Tone:** The updated body should read as a professional, well-crafted resume tailored for the desired role while incorporating the user's name and entry-level requirements.

                            The final output should be the full LaTeX document. Begin with the unaltered imports section, followed by the modified body section.
                            Only give the lateX code.

                            Here is the input resume template:

                            {resume}

                            **Instructions:**  
                            Edit only the content in the body section using the user's desired details (role, name, entry level). Ensure that the final resume is one page, uses powerful language with quantifiers, and is perfectly formatted for a high-quality professional resume.
                            Always output the full lateX body including all section.
                            Once done, output the complete LaTeX file (imports section unchanged, body section updated).

                            ---

                            This prompt should give the LLM a solid direction to produce the edited LaTeX resume per the user's requirements. Let me know if you need further tweaks or additional instructions!
                            '''
        self.initial_template = EDITOR_PROMPT.format(
            role_level=self.form_data['role_level'],
            name = self.form_data['user_name'],
            desired_role=self.form_data['desired_role'],
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
        form_data = json.load(file)
    while(True):  
        with open("main.tex", "r", encoding="utf-8") as tex_file:
            resume_data = tex_file.read()
        resume = resume_data.replace("{", "{{").replace("}", "}}")
        obj_guide = Build(resume=resume,form_data=form_data)
        query = input("\n\nEnter Query ---->\n\n")
        response = obj_guide.invoke(query= str(query))
        cropped = crop_with_tags_and_overwrite("main.tex",content = str(response))
        print(cropped) 