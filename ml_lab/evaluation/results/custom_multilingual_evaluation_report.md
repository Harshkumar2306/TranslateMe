# Scientific Evaluation Report: Custom Multilingual Transformer

Evaluated on independent held-out 4-way test records (4 parallel records × 12 directions = 48 evaluations, 0% training overlap).

- **Active Parameters**: 113,292,800 (113.29M)
- **Device**: `mps`
- **Mean SacreBLEU**: 10.26 | **Mean chrF++**: 43.71
- **Evaluated At**: 2026-11-27T06:24:00.831941+00:00

| Direction | SacreBLEU | chrF++ | Avg Latency (s) | Throughput (c/s) | Empty Rate | Repetition Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **en -> hi** | 11.17 | 47.08 | 1.399s | 47.9 | 0.0% | 0.0% |
| **en -> ta** | 4.91 | 39.74 | 1.994s | 33.6 | 0.0% | 0.0% |
| **en -> te** | 4.71 | 40.31 | 1.926s | 34.8 | 0.0% | 0.0% |
| **hi -> en** | 34.22 | 65.19 | 0.261s | 264.2 | 0.0% | 0.0% |
| **hi -> ta** | 8.2 | 45.02 | 1.946s | 35.5 | 0.0% | 0.0% |
| **hi -> te** | 4.6 | 36.33 | 1.862s | 37.1 | 0.0% | 0.0% |
| **ta -> en** | 18.47 | 58.29 | 0.269s | 286.4 | 0.0% | 0.0% |
| **ta -> hi** | 4.45 | 30.71 | 1.601s | 48.1 | 0.0% | 0.0% |
| **ta -> te** | 3.61 | 31.31 | 1.634s | 47.1 | 0.0% | 0.0% |
| **te -> en** | 14.49 | 49.61 | 0.29s | 259.9 | 0.0% | 0.0% |
| **te -> hi** | 9.79 | 40.07 | 1.641s | 46.0 | 0.0% | 0.0% |
| **te -> ta** | 4.49 | 40.86 | 1.939s | 38.9 | 0.0% | 0.0% |


### Sample Predictions

#### en -> hi
- **Sample 1**:
  - Source: `Natural language processing analyzes syntax and semantics of text.`
  - Reference: `प्राकृतिक भाषा प्रसंस्करण पाठ के वाक्यविन्यास और शब्दार्थ का विश्लेषण करता है।`
  - Hypothesis: `प्राकृतिक भाषा प्रसंस्करण में पाठ के सिंटैक्स और अर्ध-विषयों का विश्लेषण होता है।`
- **Sample 2**:
  - Source: `Deep learning has revolutionized image recognition and speech processing.`
  - Reference: `डीप लर्निंग ने छवि पहचान और वाक् प्रसंस्करण में क्रांति ला दी है।`
  - Hypothesis: `गहरी शिक्षा ने छवि मान्यता और भाषण प्रसंस्करण को क्रांति दी है।`
#### en -> ta
- **Sample 1**:
  - Source: `Natural language processing analyzes syntax and semantics of text.`
  - Reference: `இயற்கை மொழி செயலாக்கம் உரையின் தொடரியல் மற்றும் சொற்பொருளை பகுப்பாய்வு செய்கிறது.`
  - Hypothesis: `இயற்கை மொழி பதப்படுத்துதல், உரையின் தொடர்பான செயற்போக்கு மற்றும் கருவிகளை ஆய்வு செய்கிறது.`
- **Sample 2**:
  - Source: `Deep learning has revolutionized image recognition and speech processing.`
  - Reference: `ஆழமான கற்றல் பட அங்கீகாரம் மற்றும் பேச்சு செயலாக்கத்தில் புரட்சியை ஏற்படுத்தியுள்ளது.`
  - Hypothesis: `ஆழ்ந்த கற்றல் புரட்சியை உருவாக்கியுள்ளது.`
#### en -> te
- **Sample 1**:
  - Source: `Natural language processing analyzes syntax and semantics of text.`
  - Reference: `సహజ భాషా ప్రాసెసింగ్ టెక్స్ట్ యొక్క వాక్యనిర్మాణం మరియు అర్థాన్ని విశ్లేషిస్తుంది.`
  - Hypothesis: `సహజ భాష ప్రాసెసింగ్ పాఠ్యాంశం యొక్క సిన్టాక్స్, సెమెంటిక్ విశ్లేషణ.`
- **Sample 2**:
  - Source: `Deep learning has revolutionized image recognition and speech processing.`
  - Reference: `డీప్ లెర్నింగ్ ఇమేజ్ రికగ్నిషన్ మరియు స్పీచ్ ప్రాసెసింగ్‌లో విప్లవాత్మక మార్పులు తెచ్చింది.`
  - Hypothesis: `దీనిపై లోతుగా నేర్చుకోవడం వల్ల ప్రతిబింబ గుర్తింపు మరియు వాక్స్ ప్రాసెసింగ్ విప్లవాన్ని సృష్టించింది.`
#### hi -> en
- **Sample 1**:
  - Source: `प्राकृतिक भाषा प्रसंस्करण पाठ के वाक्यविन्यास और शब्दार्थ का विश्लेषण करता है।`
  - Reference: `Natural language processing analyzes syntax and semantics of text.`
  - Hypothesis: `Natural language analyzes the phrase and word of processing text.`
- **Sample 2**:
  - Source: `डीप लर्निंग ने छवि पहचान और वाक् प्रसंस्करण में क्रांति ला दी है।`
  - Reference: `Deep learning has revolutionized image recognition and speech processing.`
  - Hypothesis: `Deep Learning has revolutionized image identity and speech processing.`
#### hi -> ta
- **Sample 1**:
  - Source: `प्राकृतिक भाषा प्रसंस्करण पाठ के वाक्यविन्यास और शब्दार्थ का विश्लेषण करता है।`
  - Reference: `இயற்கை மொழி செயலாக்கம் உரையின் தொடரியல் மற்றும் சொற்பொருளை பகுப்பாய்வு செய்கிறது.`
  - Hypothesis: `இயற்கை மொழி பதப்படுத்துதல் உரையின் சொற்பொருள் மற்றும் வார்த்தவர்களை பகுப்பாய்வு செய்கிறது.`
- **Sample 2**:
  - Source: `डीप लर्निंग ने छवि पहचान और वाक् प्रसंस्करण में क्रांति ला दी है।`
  - Reference: `ஆழமான கற்றல் பட அங்கீகாரம் மற்றும் பேச்சு செயலாக்கத்தில் புரட்சியை ஏற்படுத்தியுள்ளது.`
  - Hypothesis: `தீப் கற்றல், புகைப்படம் அடையாளம் மற்றும் வார்த்தை செயல்பாட்டில் புரட்சி கொண்டது.`
#### hi -> te
- **Sample 1**:
  - Source: `प्राकृतिक भाषा प्रसंस्करण पाठ के वाक्यविन्यास और शब्दार्थ का विश्लेषण करता है।`
  - Reference: `సహజ భాషా ప్రాసెసింగ్ టెక్స్ట్ యొక్క వాక్యనిర్మాణం మరియు అర్థాన్ని విశ్లేషిస్తుంది.`
  - Hypothesis: `ప్రకృతి భాష ప్రోసెసింగ్ పాఠ్యం యొక్క సాక్ష్యము మరియు పదములను విశ్లేషిస్తుంది.`
- **Sample 2**:
  - Source: `डीप लर्निंग ने छवि पहचान और वाक् प्रसंस्करण में क्रांति ला दी है।`
  - Reference: `డీప్ లెర్నింగ్ ఇమేజ్ రికగ్నిషన్ మరియు స్పీచ్ ప్రాసెసింగ్‌లో విప్లవాత్మక మార్పులు తెచ్చింది.`
  - Hypothesis: `దీప్ లెర్నింగ్ చిత్రం గుర్తింపు మరియు మాటల ప్రక్రియలో విప్లవాన్ని సృష్టించింది.`
#### ta -> en
- **Sample 1**:
  - Source: `இயற்கை மொழி செயலாக்கம் உரையின் தொடரியல் மற்றும் சொற்பொருளை பகுப்பாய்வு செய்கிறது.`
  - Reference: `Natural language processing analyzes syntax and semantics of text.`
  - Hypothesis: `Natural language process analyzes the text's serial and vocabulary.`
- **Sample 2**:
  - Source: `ஆழமான கற்றல் பட அங்கீகாரம் மற்றும் பேச்சு செயலாக்கத்தில் புரட்சியை ஏற்படுத்தியுள்ளது.`
  - Reference: `Deep learning has revolutionized image recognition and speech processing.`
  - Hypothesis: `The deep learning has revolutionized the recognition and speech process.`
#### ta -> hi
- **Sample 1**:
  - Source: `இயற்கை மொழி செயலாக்கம் உரையின் தொடரியல் மற்றும் சொற்பொருளை பகுப்பாய்வு செய்கிறது.`
  - Reference: `प्राकृतिक भाषा प्रसंस्करण पाठ के वाक्यविन्यास और शब्दार्थ का विश्लेषण करता है।`
  - Hypothesis: `प्राकृतिक भाषा प्रक्रिया संवाद की विश्लेषणा करती है जो पाठ के संबंध और शब्दों का अनुसंधान करता है।`
- **Sample 2**:
  - Source: `ஆழமான கற்றல் பட அங்கீகாரம் மற்றும் பேச்சு செயலாக்கத்தில் புரட்சியை ஏற்படுத்தியுள்ளது.`
  - Reference: `डीप लर्निंग ने छवि पहचान और वाक् प्रसंस्करण में क्रांति ला दी है।`
  - Hypothesis: `लंबी शिक्षा से फिल्म को मान्यता और बातचीत प्रक्रिया में विशेषज्ञता हुई है।`
#### ta -> te
- **Sample 1**:
  - Source: `இயற்கை மொழி செயலாக்கம் உரையின் தொடரியல் மற்றும் சொற்பொருளை பகுப்பாய்வு செய்கிறது.`
  - Reference: `సహజ భాషా ప్రాసెసింగ్ టెక్స్ట్ యొక్క వాక్యనిర్మాణం మరియు అర్థాన్ని విశ్లేషిస్తుంది.`
  - Hypothesis: `ప్రకృతి భాష ప్రక్రియ పాఠ్యము యొక్క సంభాషణ మరియు పదమును విశ్లేషిస్తుంది.`
- **Sample 2**:
  - Source: `ஆழமான கற்றல் பட அங்கீகாரம் மற்றும் பேச்சு செயலாக்கத்தில் புரட்சியை ஏற்படுத்தியுள்ளது.`
  - Reference: `డీప్ లెర్నింగ్ ఇమేజ్ రికగ్నిషన్ మరియు స్పీచ్ ప్రాసెసింగ్‌లో విప్లవాత్మక మార్పులు తెచ్చింది.`
  - Hypothesis: `దీర్ఘ నేర్చుకోవడం, సినిమా గుర్తింపు మరియు మాట్లాడే ప్రక్రియలో విప్లవం ఏర్పడింది.`
#### te -> en
- **Sample 1**:
  - Source: `సహజ భాషా ప్రాసెసింగ్ టెక్స్ట్ యొక్క వాక్యనిర్మాణం మరియు అర్థాన్ని విశ్లేషిస్తుంది.`
  - Reference: `Natural language processing analyzes syntax and semantics of text.`
  - Hypothesis: `Natural language processing is analyzing the meaning and structure of text.`
- **Sample 2**:
  - Source: `డీప్ లెర్నింగ్ ఇమేజ్ రికగ్నిషన్ మరియు స్పీచ్ ప్రాసెసింగ్‌లో విప్లవాత్మక మార్పులు తెచ్చింది.`
  - Reference: `Deep learning has revolutionized image recognition and speech processing.`
  - Hypothesis: `Deep Learning Image Recognition and Speech Processing have brought revolutionary changes.`
#### te -> hi
- **Sample 1**:
  - Source: `సహజ భాషా ప్రాసెసింగ్ టెక్స్ట్ యొక్క వాక్యనిర్మాణం మరియు అర్థాన్ని విశ్లేషిస్తుంది.`
  - Reference: `प्राकृतिक भाषा प्रसंस्करण पाठ के वाक्यविन्यास और शब्दार्थ का विश्लेषण करता है।`
  - Hypothesis: `प्राकृतिक भाषा प्रसंस्करण के शब्दों का विश्लेषण करता है और इसका अर्थ है।`
- **Sample 2**:
  - Source: `డీప్ లెర్నింగ్ ఇమేజ్ రికగ్నిషన్ మరియు స్పీచ్ ప్రాసెసింగ్‌లో విప్లవాత్మక మార్పులు తెచ్చింది.`
  - Reference: `डीप लर्निंग ने छवि पहचान और वाक् प्रसंस्करण में क्रांति ला दी है।`
  - Hypothesis: `इस फोन में क्रांतिकारी बदलाव हुए हैं जिसमें दीपक सीखने वाली छवि रिग्नीशन और वक्त प्रक्रिया शामिल है।`
#### te -> ta
- **Sample 1**:
  - Source: `సహజ భాషా ప్రాసెసింగ్ టెక్స్ట్ యొక్క వాక్యనిర్మాణం మరియు అర్థాన్ని విశ్లేషిస్తుంది.`
  - Reference: `இயற்கை மொழி செயலாக்கம் உரையின் தொடரியல் மற்றும் சொற்பொருளை பகுப்பாய்வு செய்கிறது.`
  - Hypothesis: `இயற்கை மொழி பதப்படுத்தும் உரையின் சொற்பொருளையும், அர்த்தத்தையே ஆய்வு செய்கிறது.`
- **Sample 2**:
  - Source: `డీప్ లెర్నింగ్ ఇమేజ్ రికగ్నిషన్ మరియు స్పీచ్ ప్రాసెసింగ్‌లో విప్లవాత్మక మార్పులు తెచ్చింది.`
  - Reference: `ஆழமான கற்றல் பட அங்கீகாரம் மற்றும் பேச்சு செயலாக்கத்தில் புரட்சியை ஏற்படுத்தியுள்ளது.`
  - Hypothesis: `தீவிர கற்றல் பிம்பத்தை பதிவு செய்தல், தொலைபேசி செயலாக்கத்தில் புரட்சிகரமான மாற்றங்களை கொண்டுள்ளது.`
