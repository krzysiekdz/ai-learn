import PIL.Image
import google.generativeai as genai
import matplotlib.pyplot as plt

api_key = 'copy from config'

genai.configure(api_key=api_key)

# for model in genai.list_models():
#     print(model.name)

# model = genai.GenerativeModel('models/gemini-pro')
# res = model.generate_content('Napisz wiersz o Polsce, niech ironicznie chwali Polskę')
# print(res.text)
# print(res)

model = genai.GenerativeModel('models/gemini-1.5-flash')
img = PIL.Image.open('img2.jpg')
# plt.imshow(img)
res = model.generate_content(
    contents=[
        'This image is a Polish certificate, specifically a medical opinion confirming the ability to work. Answer in JSON format: date of issue as "di", ' + 
        'next date of visit as "nv", doctor name as "doc", person name as "person", person PESEL as "pesel", is positive (true or false) as "positive"', 
              img]
)
print(res.text)