import json
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List

class DiscriptionExtract:
    
    def __init__(self):
        
        DISCRIPTION_EXTRACTOR = PromptTemplate.from_template("""
                                You are an expert career advisor AI.

                                Given the following job description, extract and return a valid JSON object with the following keys:
                                - "required_skills": list of key technical and soft skills
                                - "education_level": minimum required or preferred education level
                                - "recommended_projects": list of project ideas not more than two
                                - "recommended_positions_of_responsibility": list of relevant leadership experiences not more than two

                                Respond ONLY in valid JSON format, no text outside the JSON block.

                                Job Description:
                                \"\"\"{input}\"\"\"
                                """)

        
        llm = ChatGroq(model= 'llama-3.3-70b-versatile',  groq_api_key=os.getenv("ARMAN_GROQ_API_KEY"))
        self.extract_chain = DISCRIPTION_EXTRACTOR | llm
    
    def invoke(self, discription):
        response = self.extract_chain.invoke({
            'input' : discription
        })
        return response.content
        
    
    
if __name__ == '__main__':
    with open('jobDis.txt','r') as jobDis:
        discription = jobDis.read()
    obj_dis = DiscriptionExtract()
    response = obj_dis.invoke(discription=str(discription))
    print(json.loads(str(response)[3:-3]))
    
    
        