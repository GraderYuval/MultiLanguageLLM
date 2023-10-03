from torch.utils.data import Dataset

class XNLIDataset(Dataset):
    def __init__(self, tokenizer, dataset, special_token='', input_max_length=100, target_max_length=10, target_language='english',
                 translate=False, language=None):
        self.tokenizer = tokenizer
        self.premises = dataset["premise"]
        self.hypotheses = dataset["hypothesis"]
        self.labels = dataset["label"]
        self.special_token = special_token
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length
        self.target_language = target_language
        self.translate = translate  # added translate flag
        self.language = language
        
        
    def __len__(self):
        return len(self.premises)

    def __getitem__(self, index):
        premise = self.premises[index]
        hypothesis = self.hypotheses[index]
        label = self.labels[index]

        return {
            "input_ids":
                self.tokenizer(self._create_input_text(premise, hypothesis),
                               max_length=self.input_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "labels":
                self.tokenizer(self._translate_target_text(label),
                               max_length=self.target_max_length,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")["input_ids"].flatten(),
            "premise": premise,
            "hypothesis": hypothesis
        }


    def _create_input_text(self, premise, hypothesis):
        instructions = []

        # Add translation instruction if translate is True
        if self.translate:
            instructions.append(*self._translate_translation_prompt())
            #instructions.append(f"Translate the following text into {self.target_language}. "
            #                    "Make sure your translation is accurate. "
            #                    "Then, based on the translated text, compute the task:")

        # Add main task instruction
        instructions.extend(self._translate_prompt(premise, hypothesis))
        #instructions.extend([
        #    f"{self.special_token} Please identify whether the premise entails or contradicts "
        #    "the hypothesis in the following premise and hypothesis. "
        #    "The answer should be exact \"entailment\", \"contradiction\", or \"neutral\".",
        #    f"Premise: {premise}",
        #    f"Hypothesis: {hypothesis}"
        #])

        return "\n".join(instructions)

    @staticmethod
    def _create_target_text(label):
        if label == 0:
            return "entailment"
        if label == 1:
            return "neutral"        
        if label == 2:
            return "contradiction"
        
        return

    def _translate_prompt(self, premise, hypothesis):
        if self.language is None:
            return prompt
        
        self.prompt_dict = {
            "en": [
                f"{self.special_token} Please identify whether the premise entails or contradicts "
                "the hypothesis in the following premise and hypothesis. "
                "The answer should be exact \"entailment\", \"contradiction\", or \"neutral\".",
                f"Premise: {premise}",
                f"Hypothesis: {hypothesis}"
            ],
            "es" :[
                f"{self.special_token} Por favor, identifique si la premisa implica o contradice "
                "la hipótesis en la siguiente premisa e hipótesis. "
                "La respuesta debe ser exactamente \"implicación\", \"contradicción\" o \"neutral\".",
                f"Premisa: {premise}",
                f"Hipótesis: {hypothesis}"
            ],
            "bg": [
                f"{self.special_token} Моля, идентифицирайте дали предпоставката подкрепя или противоречи "
                "на хипотезата в следната предпоставка и хипотеза. "
                "Отговорът трябва да бъде точно \"подкрепа\", \"противоречие\" или \"неутрален\".",
                f"Предпоставка: {premise}",
                f"Хипотеза: {hypothesis}"
            ],
            "fr": [
                f"{self.special_token} Veuillez identifier si la prémisse implique ou contredit "
                "l'hypothèse dans la prémisse et l'hypothèse suivantes. "
                "La réponse doit être exacte : \"implication\", \"contradiction\" ou \"neutre\".",
                f"Prémisse : {premise}",
                f"Hypothèse : {hypothesis}"
            ],
            "de": [
                f"{self.special_token} Bitte identifizieren Sie, ob die Prämisse die Hypothese in der folgenden Prämisse und Hypothese unterstützt oder widerspricht. Die Antwort sollte genau \"Unterstützung\", \"Widerspruch\" oder \"neutral\" sein.",
                f"Prämisse: {premise}",
                f"Hypothese: {hypothesis}"
            ],
            "el": [
                f"{self.special_token} Παρακαλώ αναγνωρίστε εάν η πρόταση συνεπάγεται ή αντιτίθεται "
                "στην υπόθεση στην ακόλουθη πρόταση και υπόθεση. "
                "Η απάντηση πρέπει να είναι ακριβώς \"συνεπαγωγή\", \"αντίφαση\" ή \"άκυρο\".",
                f"Πρόταση: {premise}",
                f"Υπόθεση: {hypothesis}"
            ],
            "ru": [
                f"{self.special_token} Пожалуйста, определите, подтверждает ли предпосылка или противоречит "
                "гипотезе в следующей предпосылке и гипотезе. "
                "Ответ должен быть точно \"подтверждение\", \"противоречие\" или \"нейтральное\".",
                f"Предпосылка: {premise}",
                f"Гипотеза: {hypothesis}"
            ],
            "zh": [
                f"{self.special_token} 请确定以下前提是否涵盖或与假设相矛盾。答案应该是\"包含\"、\"矛盾\"或\"中立\"。",
                f"前提：{premise}",
                f"假设：{hypothesis}"
            ],
            "ar": [
                f"{self.special_token} الرجاء التعرف على ما إذا كانت الفرضية تشمل أو تتعارض مع "
                "الافتراض في الفرضية والاستدلال التالي. "
                "يجب أن يكون الجواب بالضبط \"اشتمال\"، \"تناقض\"، أو \"محايد\".",
                f"الفرضية: {premise}",
                f"الاستدلال: {hypothesis}"
            ],
            "vi": [
                f"{self.special_token} Xin vui lòng xác định liệu tiền đề có gắn liền hoặc trái ngược "
                "với giả thuyết trong tiền đề và giả thuyết sau đây. "
                "Câu trả lời phải là \"gắn liền\", \"trái ngược\", hoặc \"trung lập\".",
                f"Tiền đề: {premise}",
                f"Giả thuyết: {hypothesis}"
            ],
            "hi": [
                f"{self.special_token} कृपया पहचानें कि क्या प्रस्तावना को किसी पूर्वप्रतिष्ठा या विरोध में लाता है "
                "निम्नलिखित पूर्वप्रतिष्ठा और पूर्वानुमान में। "
                "उत्तर बिल्कुल \"पूर्वप्रतिष्ठा\", \"विरोध\" या \"मध्यस्थ\" होना चाहिए।",
                f"पूर्वप्रतिष्ठा: {premise}",
                f"पूर्वानुमान: {hypothesis}"
            ],
            "sw": [
                f"{self.special_token} Tafadhali tambua ikiwa tathmini inaambatana au inapingana "
                "na dhana katika tathmini na tathmini inayofuata. "
                "Jibu linapaswa kuwa \"kuambatana\", \"kupingana\", au \"sio upande wowote\".",
                f"Tathmini: {premise}",
                f"Dhana: {hypothesis}"
            ],
            "th": [
                f"{self.special_token} โปรดระบุว่าสมมติฐานมีความเกี่ยวข้องหรือขัดแย้งกับ "
                "สมมติฐานในสรุปและสมมติฐานดังต่อไปนี้หรือไม่ "
                "คำตอบควรเป็น \"ความเกี่ยวข้อง\", \"ความขัดแย้ง\", หรือ \"เป็นกลาง\" อย่างแน่นอน",
                f"สมมติฐาน: {premise}",
                f"สมมติฐานรอง: {hypothesis}"
            ],
            "ur": [
                f"{self.special_token} براہ کرم تشخیص دیں کہ کیا ماخذ کو دعوی سے مطابقت ہے یا مخالفت کرتا ہے "
                "ماخذ اور دعوی کی تشخیص کے لئے. "
                "جواب براہ کرم \"مطابقت\"، \"مخالفت\"، یا \"غیر جانب دار\" ہونا چاہئے۔",
                f"ماخذ: {premise}",
                f"دعوی: {hypothesis}"
            ],
            "tr": [
                f"{self.special_token} Lütfen önermenin aşağıdaki önerme ve hipotezdeki önermeyi veya çürütmeyi belirleyin. Cevap kesin olarak \"çıkarım\", \"çelişki\" veya \"tarafsız\" olmalıdır.",
                f"Önerme: {premise}",
                f"Hipotez: {hypothesis}"
            ],
        }
        
        return self.prompt_dict[self.language]
        
        
    def _translate_target_text(self, label):
        if self.language is None:
            self._create_target_text(label)
            
        self.target_dict = {
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
        
        return self.target_dict[label][self.language]
        
    
    
    
    def _translate_translation_prompt(self):
        if self.language is None:
            return prompt
        
        self.prompt_dict = {
            "en": [
                    f"Translate the following text into {self.target_language}. "
                    "Make sure your translation is accurate. "
                    "Then, based on the translated text, compute the task:"
            ],
            "es" :[
                f"Traduzca el siguiente texto al {self.target_language}. "
                "Asegúrese de que su traducción sea precisa. "
                "Luego, basado en el texto traducido, realice la tarea:"
            ],
            "bg": [
                f"Преведете следния текст на {self.target_language}. "
                "Уверете се, че преводът ви е точен. "
                "След това, базирайте се на преведения текст и изпълнете задачата:"
            ],
            "fr": [
                f"Traduisez le texte suivant en {self.target_language}. "
                "Assurez-vous que votre traduction est précise. "
                "Ensuite, en vous basant sur le texte traduit, effectuez la tâche :"
            ],
            "de": [
                f"Übersetzen Sie den folgenden Text ins {self.target_language}. "
                "Stellen Sie sicher, dass Ihre Übersetzung korrekt ist. "
                "Dann, basierend auf dem übersetzten Text, führen Sie die Aufgabe aus:"
            ],
            "el": [
                f"Μεταφράστε τον παρακάτω κείμενο στη γλώσσα {self.target_language}. "
                "Βεβαιωθείτε ότι η μετάφρασή σας είναι ακριβής. "
                "Στη συνέχεια, με βάση το μεταφρασμένο κείμενο, εκτελέστε την εργασία:"
            ],
            "ru": [
                f"Переведите следующий текст на {self.target_language}. "
                "Убедитесь, что ваш перевод точен. "
                "Затем, основываясь на переведенном тексте, выполните задачу:"
            ],
            "zh": [
                f"将以下文本翻译成{self.target_language}。"
                "确保您的翻译准确。"
                "然后，根据翻译的文本，执行任务："
            ],
            "ar": [
                f"قم بترجمة النص التالي إلى {self.target_language}. "
                "تأكد من دقة ترجمتك. "
                "ثم، استند إلى النص المترجم لأداء المهمة:"
            ],
            "vi": [
                f"Dịch văn bản sau sang {self.target_language}. "
                "Hãy đảm bảo rằng bản dịch của bạn là chính xác. "
                "Sau đó, dựa vào văn bản đã dịch, thực hiện nhiệm vụ:"
            ],
            "hi": [
                f"निम्नलिखित पाठ का {self.target_language} में अनुवाद करें। "
                "सुनिश्चित करें कि आपका अनुवाद सटीक है। "
                "फिर, अनुवादित पाठ के आधार पर कार्य करें:"
            ],
            "sw": [
                f"Tafsiri nakala ifuatayo kwa {self.target_language}. "
                "Hakikisha tafsiri yako ni sahihi. "
                "Kisha, kulingana na nakala iliyotafsiriwa, tekeleza kazi:"
            ],
            "th": [
                f"แปลข้อความต่อไปนี้เป็น {self.target_language} โปรดตรวจสอบว่าการแปลของคุณถูกต้อง"
                "จากนั้น จากข้อความที่ถูกแปล ดำเนินการคำสั่ง:"
            ],
            "ur": [
                f"مندرجہ ذیل متن کو {self.target_language} میں ترجمہ کریں۔ "
                "یقینی بنائیں کہ آپ کا ترجمہ درست ہے۔ "
                "پھر، ترجمہ شدہ متن پر مبنی کام کریں:"
            ],
            "tr": [
                f"Aşağıdaki metni {self.target_language}'ye çevirin. "
                "Çevirinizin doğru olduğundan emin olun. "
                "Ardından, çevrilen metne dayanarak görevi gerçekleştirin:"
            ],
        }
        
        return self.prompt_dict[self.language]