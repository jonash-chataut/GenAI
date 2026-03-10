from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):

    name:str = 'Jonash' #jonash is the default value
    age: Optional[int]=None
    # email:EmailStr
    cgpa:float=Field(gt=0,lt=10,default=7)


new_student={'name':'Jonash','age':'20','cpga':5}
student=Student(**new_student)
print(student)
