from flask import Flask, request, jsonify, render_template
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from sklearn.preprocessing import LabelEncoder
import torch
import joblib
from easynertag import Engine
from pythainlp.tokenize import word_tokenize

app = Flask(__name__)

# Load model and tokenizer
model_path = './model/model_Intent'
tokenizer_path = './model/model_Intent'
label_encoder_path = './model/label_encoder.pkl'

model = CamembertForSequenceClassification.from_pretrained(model_path)
tokenizer = CamembertTokenizer.from_pretrained(tokenizer_path)
label_encoder = joblib.load(label_encoder_path)


questions = [
    "พัดลมคูลเลอร์หน้าจอปกติ แต่หน้างานจริงไม่ทำงาน เบรคเกอร์ เปิด ปกติ",
    "รบกวนสอบถามไลน์ Meal Silo “Job Setting> Bin Ident Meal Silo”จากการใช้งานพึงกลับมาจับได้ 3 วัน ทางคลังเลือก ให้ใช้งาน 2 ถัง M105/M110 ตึก 1 ลากไลน์ Meal Down TR9 M110 อยู่ แต่ ตึก 2 จะลากไลน์ MealUp TR9 M105 “แต่ job วิ่งไปจับถัง M110 ครับ  โปรแกรมได้จัด Prioiy ไว้ไหมครับ",
    "ใบเชนติดตะแกลงเครื่องอัดเม็ด",
    "น็อตขาด",
    "ค่าอุณหภูมิเครื่องคอนค้าง ไม่อัพเดต",
    "temp cooler สูงกว่าปกติ",
    "เครื่องปั้มเม็ดสอง มีรอยเหล็กลงลูกกลิ้ง นอตขาด1 ดรายมีลูกเหล็ก ลูกกลิ้งปกติ น็อตครบ",
    "stream ตก",
    "น็อตลงดราย",
    "ตะแกลงร่อนขาด ชั้นกลางฝั่ง boiler",
    "ไลน์สอง แม่เหล็ก ลงคอน หลุด"
]

@app.route("/", methods=["GET", "POST"])
def index():
    selected_question = None
    user_question = None
    predicted_intent = None
    replaced_text = None
    entity_recognition_results = None
    
    patterns = {
        "Hammer mill": "Hammer,เครื่องบด,grinder",
        "Mixer": "เครื่องผสมวัตุดิบ,มิกซ์,เครื่องผสม,มิกซ์เซอร์",
        "Pellet mill": "เครื่องอัดเม็ด,ปั๊มเม็ด,เครื่องขึ้นรูป,เครื่องปั้มเม็ด",
        "Conditioner": "เครื่องคอน",
        "Cooler": "คูลเลอร์,พัดลม,cooler",
        "Boiler": "หม้อไอน้ำ,boiler",
        "Chain": "ใบเชน",
        "Drier": "เครื่องอบ,ดราย",
        "Silo": "ไซโล,คลัง,Silo",
        "Conveyor": "ลำเลียง",
        "Scada": "หน้าจอ Scada",
        "Temperature": "อุณหภูมิ,temp",
        "Steam": "สตรีม,ไอน้ำ",
        "Breaker": "เบรคเกอร์",
        "bucket belt": "กระพ้อ",
        "Silo Car": "รถไซโล",
        "Extruder": "เครื่องนึ่ง",
        "hopper": "ถัง",
        "scale": "เครื่องชั่ง",
        "package": "แพคเกจ",
        "expander": "เครื่องนวด",
        "grat": "ตะแกลง,เกรท"
    }


    if request.method == "POST":
        selected_question = request.form.get("question")
        user_question = request.form.get("user_question", "")

        if user_question:
            selected_question = user_question

        # Perform intent classification
        inputs = tokenizer(selected_question, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits).item()
        predicted_intent = label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Replace keywords with patterns
        replaced_text = selected_question
        for pattern, keywords in patterns.items():
            for keyword in keywords.split(','):
                replaced_text = replaced_text.replace(keyword, f"[{pattern}]{keyword}[/{pattern}]")

    return render_template("index.html", questions=questions, selected_question=selected_question, predicted_intent=predicted_intent, replaced_text=replaced_text, entity_recognition_results=entity_recognition_results, patterns=patterns)


@app.route("/find_patterns", methods=["POST"])
def find_patterns():
    try:
        selected_question = request.form["question"]
        replaced_text = selected_question

        for pattern, keywords in patterns.items():
            for keyword in keywords.split(','):
                replaced_text = replaced_text.replace(keyword, f"[{pattern}]{keyword}[/{pattern}]")

        return jsonify({"original_question": selected_question, "replaced_text": replaced_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/find_entity", methods=["POST"])
def find_entity():
    try:
        data = request.get_json()
        replaced_text = data.get("replaced_text", "")

        print(replaced_text)

        if not replaced_text:
            return jsonify({"error": "No 'replaced_text' found in request."}), 400

        mylist = [replaced_text]

        print(mylist)

        word_tokenize(mylist, engine='attacut')
        build = Engine(word_tokenize)
        conll2002_list = []
        
        for i in mylist:
            conll2002_list.append(build.text2conll2002(i))
        
        conll2002_output = '\n'.join(conll2002_list)

        print("ผลลัพธ์:", conll2002_output)

        data = []
        for line in conll2002_output.split("\n"):
            word_label = line.split()
            if len(word_label) == 2:
                word, label = word_label
                data.append([word, label])
            elif len(word_label) == 1 and word_label[0] == 'O':
                data.append(['', 'O'])

        for item in data:
            text = item
            print(text)

        return jsonify({"conll2002_output": conll2002_output,"text":text})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
