import requests
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import json

CHROMA_PATH = "chroma"
url = 'https://vmotz4cb225njh-8000.proxy.runpod.net/v1/completions'

PROMPT_TEMPLATE = """
Forget yourself, now you are a senior lawyer who can interpret the following contexts and laws.
Use this definition to Answer the question
\nโดยทุจริต = เพื่อแสวงหาประโยชน์ที่มิควรได้โดยชอบด้วยกฎหมายสําหรับตนเองหรือผู้อื่น\nทางสาธารณะ = ทางบกหรือทางน้ําสําหรับประชาชนใช้ในการจราจร และให้หมายความรวมถึงทางรถไฟและทางรถรางที่มีรถเดิน สําหรับประชาชนโดยสารด้วย\nสาธารณสถาน = สถานที่ใด ๆ ซึ่งประชาชนมีความชอบธรรมที่จะเข้าไปได้\nเคหสถาน = ที่ซึ่งใช้เป็นที่อยู่อาศัย เช่น เรือน โรง เรือ หรือแพซึ่งคนอยู่อาศัยและให้หมายความรวมถึงบริเวณของที่ซึ่งใช้เป็นที่อยู่อาศัยนั้นด้วย จะมีรั้วล้อมหรือไม่ก็ตาม\nอาวุธ = หมายความรวมถึงสิ่งซึ่งไม่เป็นอาวุธโดยสภาพ แต่ซึ่งได้ใช้หรือเจตนาจะใช้ประทุษร้ายร่างกายถึงอันตรายสาหัสอย่างอาวุธ\nใช้กําลังประทุษร้าย = ทําการประทุษร้ายแก่กายหรือจิตใจของบุคคล ไม่ว่าจะทําด้วยใช้แรงกายภาพหรือด้วยวิธีอื่นใด และให้หมายความรวมถึงการกระทําใด ๆ ซึ่งเป็นเหตุให้บุคคลหนึ่งบุคคลใดอยู่ในภาวะที่ไม่สามารถขัดขืนได้ \nไม่ว่าจะโดยใช้ยาทําให้มึนเมา สะกดจิต หรือใช้วิธีอื่นใดอันคล้ายคลึงกัน\nเอกสาร = กระดาษหรือวัตถุอื่นใดซึ่งได้ทําให้ปรากฏความหมายด้วยตัวอักษร ตัวเลข ผัง หรือแผนแบบอย่างอื่น จะเป็นโดยวิธีพิมพ์ ถ่ายภาพ หรือวิธีอื่นอันเป็นหลักฐานแห่งความหมายนั้น\nเอกสารราชการ = เอกสารซึ่งเจ้าพนักงานได้ทําขึ้นหรือรับรองในหน้าที่ และให้หมายความรวมถึงสําเนาเอกสารนั้น ๆ ที่เจ้าพนักงานได้รับรองในหน้าที่ด้วย\nเอกสารสิทธิ = เอกสารที่เป็นหลักฐานแห่งการก่อ เปลี่ยนแปลงโอน สงวนหรือระงับซึ่งสิทธิ\nลายมือชื่อ = หมายความรวมถึงลายพิมพ์นิ้วมือและเครื่องหมายซึ่งบุคคลลงไว้แทนลายมือชื่อของตน\nกลางคืน = เวลาระหว่างพระอาทิตย์ตกและพระอาทิตย์ขึ้น\nคุมขัง = คุมตัว ควบคุม ขัง กักขังหรือจําคุก\nค่าไถ่ = ทรัพย์สินหรือประโยชน์ที่เรียกเอา หรือให้เพื่อแลกเปลี่ยนเสรีภาพของผู้ถูกเอาตัวไป ผู้ถูกหน่วงเหนี่ยวหรือผู้ถูกกักขัง\nบัตรอิเล็กทรอนิกส์ หมายถึง \n(ก)เอกสารหรือวัตถุอื่นใดไม่ว่าจะมีรูปลักษณะใดที่ผู้ออกได้ออกให้แก่ผู้มีสิทธิใช้ ซึ่งจะระบุชื่อหรือไม่ก็ตาม โดยบันทึกข้อมูลหรือรหัสไว้ด้วยการประยุกต์ใช้วิธีการทางอิเล็กตรอนไฟฟ้า คลื่นแม่เหล็กไฟฟ้า \nหรือวิธีอื่นใดในลักษณะคล้ายกัน ซึ่งรวมถึงการประยุกต์ใช้วิธีการทางแสงหรือวิธีการทางแม่เหล็กให้ปรากฏความหมายด้วยตัวอักษร ตัวเลข รหัส หมายเลขบัตร หรือสัญลักษณ์อื่นใด ทั้งที่สามารถมองเห็นและมองไม่เห็นด้วยตาเปล่า\n(ข)ข้อมูล รหัส หมายเลขบัญชี หมายเลขชุดทางอิเล็กทรอนิกส์หรือเครื่องมือทางตัวเลขใด ๆ ที่ผู้ออกได้ออกให้แก่ผู้มีสิทธิใช้ โดยมิได้มีการออกเอกสารหรือวัตถุอื่นใดให้ แต่มีวิธีการใช้ในทํานองเดียวกับ (ก) หรือ\n(ค)สิ่งอื่นใดที่ใช้ประกอบกับข้อมูลอิเล็กทรอนิกส์เพื่อแสดงความสัมพันธ์ระหว่างบุคคลกับข้อมูลอิเล็กทรอนิกส์ โดยมีวัตถุประสงค์เพื่อระบุตัวบุคคลผู้เป็นเจ้าของ\nหนังสือเดินทาง = เอกสารสําคัญประจําตัวไม่ว่าจะมีรูปลักษณะใดที่รัฐบาลไทย รัฐบาลต่างประเทศ หรือองค์การระหว่างประเทศออกให้แก่บุคคลใด เพื่อใช้แสดงตนในการเดินทางระหว่างประเทศ \nและให้หมายความรวมถึงเอกสารใช้แทนหนังสือเดินทางและแบบหนังสือเดินทางที่ยังไม่ได้กรอกข้อความเกี่ยวกับผู้ถือหนังสือเดินทางด้วย\nเจ้าพนักงาน = บุคคลซึ่งกฎหมายบัญญัติว่าเป็นเจ้าพนักงานหรือได้รับแต่งตั้งตามกฎหมายให้ปฏิบัติหน้าที่ราชการ ไม่ว่าเป็นประจําหรือครั้งคราว และไม่ว่าจะได้รับค่าตอบแทนหรือไม่\nสื่อลามกอนาจารเด็ก = วัตถุหรือสิ่งที่แสดงให้รู้หรือเห็นถึงการกระทําทางเพศของเด็กหรือกับเด็กซึ่งมีอายุไม่เกินสิบแปดปี โดยรูป เรื่อง หรือลักษณะสามารถสื่อไปในทางลามกอนาจาร ไม่ว่าจะอยู่ในรูปแบบของเอกสาร ภาพเขียน ภาพพิมพ์ ภาพระบายสี สิ่งพิมพ์\nรูปภาพ ภาพโฆษณา เครื่องหมาย รูปถ่าย ภาพยนตร์ แถบบันทึกเสียง แถบบันทึกภาพ หรือรูปแบบอื่นใดในลักษณะทํานองเดียวกัน และให้หมายความรวมถึงวัตถุหรือสิ่งต่าง ๆ ข้างต้นที่จัดเก็บในระบบคอมพิวเตอร์หรือในอุปกรณ์อิเล็กทรอนิกส์อื่นที่สามารถแสดงผลให้เข้าใจความหมายได้\n
{prompt}
Answer the question with Thai language adapted to the following context:
{context}

Question : {question}
"""


def arabic_to_thai_number(input_str: str) -> str:
    thai_digits = str.maketrans("0123456789", "๐๑๒๓๔๕๖๗๘๙")
    return input_str.translate(thai_digits)

def query_vllm(prompt: str):
    payload = {
        "model": "scb10x/llama3.1-typhoon2-8b-instruct",
        "prompt": prompt,
        "max_tokens": 450,
        "temperature": 0.6,
        "top_p": 0.9,
        "frequency_penalty": 0.8,
        "presence_penalty": 0.4,
        "stream": False
    }


    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"
    
def loadJSONfile(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        data = json.load(file)
        return data


def response(question):
    query_text = question
    query_text = arabic_to_thai_number(query_text)
    embedding_function = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    # print(f"Query: {query_text}")
    # print(f"Results: {results}")
    
    if len(results) == 0 or results[0][1] < 0.75:
        print(f"Unable to find matching results.")
        return

    files = set()
    prompt_texts = []
    context_texts = []
    data = loadJSONfile("mapping_chunk.json")
    mapping_prompt = loadJSONfile("mapping_prompt.json")
    for doc, score in results:
        chunk_number = doc.metadata["chunk_number"]
        # print(f"chunk_number : {chunk_number}")
        for value in data:
            if chunk_number >= data[value][0] and chunk_number <= data[value][1]:
                files.add(value)
                break
            else:
                continue

        context_texts.append(f"{doc.page_content}")

    for value in files:
        prompt_texts.append(f"{mapping_prompt[value]}")        

    context_text = "".join(context_texts)

    prompt_text = "".join(prompt_texts)
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question, prompt=prompt_text)

    print(prompt)

    response_text = query_vllm(prompt)

    if isinstance(response_text, dict) and 'choices' in response_text:
        return response_text['choices'][0]['text']
    else:
        return response_text  # if it's already a string

