def build_xnli_prompt(special_token, premise, hypothesis, language):
    prompt_dict = {
        "en": [
            f"{special_token} Please identify whether the premise entails or contradicts "
            "the hypothesis in the following premise and hypothesis. "
            "The answer should be exact \"entailment\", \"contradiction\", or \"neutral\".",
            f"Premise: {premise}",
            f"Hypothesis: {hypothesis}"
        ],
        "es" :[
            f"{special_token} Por favor, identifique si la premisa implica o contradice "
            "la hipótesis en la siguiente premisa e hipótesis. "
            "La respuesta debe ser exactamente \"implicación\", \"contradicción\" o \"neutral\".",
            f"Premisa: {premise}",
            f"Hipótesis: {hypothesis}"
        ],
        "bg": [
            f"{special_token} Моля, идентифицирайте дали предпоставката подкрепя или противоречи "
            "на хипотезата в следната предпоставка и хипотеза. "
            "Отговорът трябва да бъде точно \"подкрепа\", \"противоречие\" или \"неутрален\".",
            f"Предпоставка: {premise}",
            f"Хипотеза: {hypothesis}"
        ],
        "fr": [
            f"{special_token} Veuillez identifier si la prémisse implique ou contredit "
            "l'hypothèse dans la prémisse et l'hypothèse suivantes. "
            "La réponse doit être exacte : \"implication\", \"contradiction\" ou \"neutre\".",
            f"Prémisse : {premise}",
            f"Hypothèse : {hypothesis}"
        ],
        "de": [
            f"{special_token} Bitte identifizieren Sie, ob die Prämisse die Hypothese in der folgenden Prämisse und Hypothese unterstützt oder widerspricht. Die Antwort sollte genau \"Unterstützung\", \"Widerspruch\" oder \"neutral\" sein.",
            f"Prämisse: {premise}",
            f"Hypothese: {hypothesis}"
        ],
        "el": [
            f"{special_token} Παρακαλώ αναγνωρίστε εάν η πρόταση συνεπάγεται ή αντιτίθεται "
            "στην υπόθεση στην ακόλουθη πρόταση και υπόθεση. "
            "Η απάντηση πρέπει να είναι ακριβώς \"συνεπαγωγή\", \"αντίφαση\" ή \"άκυρο\".",
            f"Πρόταση: {premise}",
            f"Υπόθεση: {hypothesis}"
        ],
        "ru": [
            f"{special_token} Пожалуйста, определите, подтверждает ли предпосылка или противоречит "
            "гипотезе в следующей предпосылке и гипотезе. "
            "Ответ должен быть точно \"подтверждение\", \"противоречие\" или \"нейтральное\".",
            f"Предпосылка: {premise}",
            f"Гипотеза: {hypothesis}"
        ],
        "zh": [
            f"{special_token} 请确定以下前提是否涵盖或与假设相矛盾。答案应该是\"包含\"、\"矛盾\"或\"中立\"。",
            f"前提：{premise}",
            f"假设：{hypothesis}"
        ],
        "ar": [
            f"{special_token} الرجاء التعرف على ما إذا كانت الفرضية تشمل أو تتعارض مع "
            "الافتراض في الفرضية والاستدلال التالي. "
            "يجب أن يكون الجواب بالضبط \"اشتمال\"، \"تناقض\"، أو \"محايد\".",
            f"الفرضية: {premise}",
            f"الاستدلال: {hypothesis}"
        ],
        "vi": [
            f"{special_token} Xin vui lòng xác định liệu tiền đề có gắn liền hoặc trái ngược "
            "với giả thuyết trong tiền đề và giả thuyết sau đây. "
            "Câu trả lời phải là \"gắn liền\", \"trái ngược\", hoặc \"trung lập\".",
            f"Tiền đề: {premise}",
            f"Giả thuyết: {hypothesis}"
        ],
        "hi": [
            f"{special_token} कृपया पहचानें कि क्या प्रस्तावना को किसी पूर्वप्रतिष्ठा या विरोध में लाता है "
            "निम्नलिखित पूर्वप्रतिष्ठा और पूर्वानुमान में। "
            "उत्तर बिल्कुल \"पूर्वप्रतिष्ठा\", \"विरोध\" या \"मध्यस्थ\" होना चाहिए।",
            f"पूर्वप्रतिष्ठा: {premise}",
            f"पूर्वानुमान: {hypothesis}"
        ],
        "sw": [
            f"{special_token} Tafadhali tambua ikiwa tathmini inaambatana au inapingana "
            "na dhana katika tathmini na tathmini inayofuata. "
            "Jibu linapaswa kuwa \"kuambatana\", \"kupingana\", au \"sio upande wowote\".",
            f"Tathmini: {premise}",
            f"Dhana: {hypothesis}"
        ],
        "th": [
            f"{special_token} โปรดระบุว่าสมมติฐานมีความเกี่ยวข้องหรือขัดแย้งกับ "
            "สมมติฐานในสรุปและสมมติฐานดังต่อไปนี้หรือไม่ "
            "คำตอบควรเป็น \"ความเกี่ยวข้อง\", \"ความขัดแย้ง\", หรือ \"เป็นกลาง\" อย่างแน่นอน",
            f"สมมติฐาน: {premise}",
            f"สมมติฐานรอง: {hypothesis}"
        ],
        "ur": [
            f"{special_token} براہ کرم تشخیص دیں کہ کیا ماخذ کو دعوی سے مطابقت ہے یا مخالفت کرتا ہے "
            "ماخذ اور دعوی کی تشخیص کے لئے. "
            "جواب براہ کرم \"مطابقت\"، \"مخالفت\"، یا \"غیر جانب دار\" ہونا چاہئے۔",
            f"ماخذ: {premise}",
            f"دعوی: {hypothesis}"
        ],
        "tr": [
            f"{special_token} Lütfen önermenin aşağıdaki önerme ve hipotezdeki önermeyi veya çürütmeyi belirleyin. Cevap kesin olarak \"çıkarım\", \"çelişki\" veya \"tarafsız\" olmalıdır.",
            f"Önerme: {premise}",
            f"Hipotez: {hypothesis}"
        ],
    }
    if language is None:
        return prompt_dict["en"]
    return prompt_dict[language]

def build_xnli_target(label, language):
    target_dict = {
        0: {
            "ru": "влечение",
            "vi": "kéo theo",
            "bg": "последствие",
            "es": "consecuencia",
            "de": "Folge",
            "fr": "conséquence",
            "el": "συνέπεια",
            "ar": "استتباع",
            "en": "entailment",
            "hi": "परिणति",
            "zh": "蕴涵",
            "sw": "kuambatana",
            "th": "การแสดง",
            "ur": "پرنطی",
            "tr": "sonuç"
        },
        1: {
            "ru": "нейтральный",
            "vi": "trung lập",
            "bg": "неутрален",
            "es": "neutral",
            "de": "neutral",
            "fr": "neutre",
            "el": "ουδέτερος",
            "ar": "محايد",
            "en": "neutral",
            "hi": "तटस्थ",
            "zh": "中性",
            "sw": "kutopendelea upande wowote",
            "th": "เป็นกลาง",
            "ur": "خود طرفہ",
            "tr": "tarafsız"
        },
        2: {
            "ru": "противоречие",
            "vi": "mâu thuẫn",
            "bg": "противоречие",
            "es": "contradicción",
            "de": "Widerspruch",
            "fr": "contradiction",
            "el": "αντίφαση",
            "ar": "تناقض",
            "en": "contradiction",
            "hi": "विरोध",
            "zh": "矛盾",
            "sw": "kupingana",
            "th": "ขัดแย้ง",
            "ur": "تضاد",
            "tr": "çelişki"
        }
    }
    if language is None:
        return target_dict[label]["en"]
    return target_dict[label][language]
    
def build_xquad_prompt(special_token, context, question, language):
    prompt_dict = {
        "en": [
            f"{special_token} Please answer the question according to the context in the following context and question.",
            f"Context: {context}",
            f"Question: {question}"
        ],
        "ar": [
            f"{special_token} .من فضلك، قم بالإجابة على السؤال وفقًا للسياق في السياق التالي والسؤال التالي",
            f"{context} :السياق",
            f"{question} :السؤال"
        ],
        "de": [
            f"{special_token} Bitte beantworten Sie die Frage gemäß dem Kontext im folgenden Zusammenhang und der folgenden Frage.",
            f"Kontext: {context}",
            f"Frage: {question}"
        ],
        "el": [
            f"{special_token} Παρακαλώ απαντήστε στην ερώτηση σύμφωνα με τον περιβάλλοντα χώρο στον ακόλουθο πλαίσιο και την ερώτηση.",
            f"Πλαίσιο: {context}",
            f"Ερώτηση: {question}"
        ],
        "es": [
            f"{special_token} Por favor, responde a la pregunta de acuerdo al contexto en el siguiente contexto y pregunta.",
            f"Contexto: {context}",
            f"Pregunta: {question}"
        ],
        "hi": [
            f"{special_token} कृपया निम्नलिखित संदर्भ और प्रश्न के अनुसार प्रश्न का उत्तर दें।",
            f"संदर्भ: {context}",
            f"प्रश्न: {question}"
        ],
        "ru": [
            f"{special_token} Пожалуйста, ответьте на вопрос в соответствии с контекстом в следующем контексте и вопросе.",
            f"Контекст: {context}",
            f"Вопрос: {question}"
        ],
        "th": [
            f"{special_token} โปรดตอบคำถามตามบริบทในบริบทและคำถามต่อไปนี้.",
            f"บริบท: {context}",
            f"คำถาม: {question}"
        ],
        "tr": [
            f"{special_token} Lütfen aşağıdaki bağlam ve soruya göre soruyu yanıtlayın.",
            f"Bağlam: {context}",
            f"Soru: {question}"
        ],
        "vi": [
            f"{special_token} Vui lòng trả lời câu hỏi theo ngữ cảnh trong ngữ cảnh và câu hỏi sau đây.",
            f"Ngữ cảnh: {context}",
            f"Câu hỏi: {question}"
        ],
        "zh": [
            f"{special_token} 请根据以下的背景和问题回答问题。",
            f"背景: {context}",
            f"问题: {question}"
        ],
        "ro": [
            f"{special_token} Vă rugăm să răspundeți la întrebare în funcție de contextul din următorul context și întrebare.",
            f"Context: {context}",
            f"Întrebare: {question}"
        ],
    }
    if language is None:
        return prompt_dict["en"]
    return prompt_dict[language]
    
def build_translation_prompt(target_language, language):
    prompt_dict = {
        "en": 
            f"Translate the following text into {target_language}. "
            "Make sure your translation is accurate. "
            "Then, based on the translated text, compute the task:",
        "es" :
            f"Traduzca el siguiente texto al {target_language}. "
            "Asegúrese de que su traducción sea precisa. "
            "Luego, basado en el texto traducido, realice la tarea:",
        "bg": 
            f"Преведете следния текст на {target_language}. "
            "Уверете се, че преводът ви е точен. "
            "След това, базирайте се на преведения текст и изпълнете задачата:",
        "fr": 
            f"Traduisez le texte suivant en {target_language}. "
            "Assurez-vous que votre traduction est précise. "
            "Ensuite, en vous basant sur le texte traduit, effectuez la tâche :",
        "de": 
            f"Übersetzen Sie den folgenden Text ins {target_language}. "
            "Stellen Sie sicher, dass Ihre Übersetzung korrekt ist. "
            "Dann, basierend auf dem übersetzten Text, führen Sie die Aufgabe aus:",
        "el": 
            f"Μεταφράστε τον παρακάτω κείμενο στη γλώσσα {target_language}. "
            "Βεβαιωθείτε ότι η μετάφρασή σας είναι ακριβής. "
            "Στη συνέχεια, με βάση το μεταφρασμένο κείμενο, εκτελέστε την εργασία:",
        "ru": 
            f"Переведите следующий текст на {target_language}. "
            "Убедитесь, что ваш перевод точен. "
            "Затем, основываясь на переведенном тексте, выполните задачу:",
        "zh": 
            f"将以下文本翻译成{target_language}。"
            "确保您的翻译准确。"
            "然后，根据翻译的文本，执行任务：",
        "ar": 
            f"قم بترجمة النص التالي إلى {target_language}. "
            "تأكد من دقة ترجمتك. "
            "ثم، استند إلى النص المترجم لأداء المهمة:",
        "vi": 
            f"Dịch văn bản sau sang {target_language}. "
            "Hãy đảm bảo rằng bản dịch của bạn là chính xác. "
            "Sau đó, dựa vào văn bản đã dịch, thực hiện nhiệm vụ:",
        "hi": 
            f"निम्नलिखित पाठ का {target_language} में अनुवाद करें। "
            "सुनिश्चित करें कि आपका अनुवाद सटीक है। "
            "फिर, अनुवादित पाठ के आधार पर कार्य करें:",
        "sw": 
            f"Tafsiri nakala ifuatayo kwa {target_language}. "
            "Hakikisha tafsiri yako ni sahihi. "
            "Kisha, kulingana na nakala iliyotafsiriwa, tekeleza kazi:",
        "th": 
            f"แปลข้อความต่อไปนี้เป็น {target_language} โปรดตรวจสอบว่าการแปลของคุณถูกต้อง"
            "จากนั้น จากข้อความที่ถูกแปล ดำเนินการคำสั่ง:",
        "ur": 
            f"مندرجہ ذیل متن کو {target_language} میں ترجمہ کریں۔ "
            "یقینی بنائیں کہ آپ کا ترجمہ درست ہے۔ "
            "پھر، ترجمہ شدہ متن پر مبنی کام کریں:",
        "tr": 
            f"Aşağıdaki metni {target_language}'ye çevirin. "
            "Çevirinizin doğru olduğundan emin olun. "
            "Ardından, çevrilen metne dayanarak görevi gerçekleştirin:",
    }
    if language is None:
        return prompt_dict["en"]
    return prompt_dict[language]