# Scientific Evaluation Report: Custom Multilingual Transformer

Evaluated on independent held-out test data with zero data leakage.

| Direction | SacreBLEU | chrF++ | Avg Latency (s) | Throughput (c/s) | Empty Rate | Repetition Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **en -> hi** | 0.0 | 2.58 | 1.05s | 63.8 | 0.0% | 47.4% |
| **en -> ta** | 0.58 | 8.08 | 1.425s | 47.0 | 0.0% | 48.7% |
| **en -> te** | 0.0 | 4.15 | 2.468s | 27.2 | 0.0% | 64.1% |
| **hi -> en** | 0.44 | 5.14 | 1.414s | 48.8 | 0.0% | 59.2% |
| **ta -> en** | 0.45 | 4.94 | 1.42s | 54.2 | 0.0% | 54.2% |
| **te -> en** | 0.23 | 5.21 | 2.351s | 32.1 | 0.0% | 79.5% |


### Sample Predictions

#### en -> hi
- **Sample 1**:
  - Source: `Natural language processing analyzes syntax and semantics of text.`
  - Reference: `प्राकृतिक भाषा प्रसंस्करण पाठ के वाक्यविन्यास और शब्दार्थ का विश्लेषण करता है।`
  - Hypothesis: `गण ading మూల మూల వాక్య से से से से से से से से से से से से से से से से से से से से से से से से से से से से से से से से से से से`
- **Sample 2**:
  - Source: `Deep learning has revolutionized image recognition and speech processing.`
  - Reference: `डीप लर्निंग ने छवि पहचान और वाक् प्रसंस्करण में क्रांति ला दी है।`
  - Hypothesis: `ित , நா ।`
#### en -> ta
- **Sample 1**:
  - Source: `Natural language processing analyzes syntax and semantics of text.`
  - Reference: `இயற்கை மொழி செயலாக்கம் உரையின் தொடரியல் மற்றும் சொற்பொருளை பகுப்பாய்வு செய்கிறது.`
  - Hypothesis: `புத்தகங்கள மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும்`
- **Sample 2**:
  - Source: `Deep learning has revolutionized image recognition and speech processing.`
  - Reference: `ஆழமான கற்றல் பட அங்கீகாரம் மற்றும் பேச்சு செயலாக்கத்தில் புரட்சியை ஏற்படுத்தியுள்ளது.`
  - Hypothesis: `புத்தகங்கள மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும் மற்றும்`
#### en -> te
- **Sample 1**:
  - Source: `Natural language processing analyzes syntax and semantics of text.`
  - Reference: `సహజ భాషా ప్రాసెసింగ్ టెక్స్ట్ యొక్క వాక్యనిర్మాణం మరియు అర్థాన్ని విశ్లేషిస్తుంది.`
  - Hypothesis: `ఎన్‌ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ మూల అనువాద అభ శ్రేయస్సు శ్రేయస్సు ను శ్రేయస్సు ను ను ను ను ను ను ను ను ను ను ను ను ను శ్రేయస్సు ను ను ను శ్రేయస్సు ను ను ను ను ను ను ను ను`
- **Sample 2**:
  - Source: `Deep learning has revolutionized image recognition and speech processing.`
  - Reference: `డీప్ లెర్నింగ్ ఇమేజ్ రికగ్నిషన్ మరియు స్పీచ్ ప్రాసెసింగ్‌లో విప్లవాత్మక మార్పులు తెచ్చింది.`
  - Hypothesis: `ఎన్‌ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ మూల కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ కోడర్ మూల`
#### hi -> en
- **Sample 1**:
  - Source: `प्राकृतिक भाषा प्रसंस्करण पाठ के वाक्यविन्यास और शब्दार्थ का विश्लेषण करता है।`
  - Reference: `Natural language processing analyzes syntax and semantics of text.`
  - Hypothesis: `The er generat es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es`
- **Sample 2**:
  - Source: `डीप लर्निंग ने छवि पहचान और वाक् प्रसंस्करण में क्रांति ला दी है।`
  - Reference: `Deep learning has revolutionized image recognition and speech processing.`
  - Hypothesis: `Than k ver y y y much kin help .`
#### ta -> en
- **Sample 1**:
  - Source: `இயற்கை மொழி செயலாக்கம் உரையின் தொடரியல் மற்றும் சொற்பொருளை பகுப்பாய்வு செய்கிறது.`
  - Reference: `Natural language processing analyzes syntax and semantics of text.`
  - Hypothesis: `The er generat es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es`
- **Sample 2**:
  - Source: `ஆழமான கற்றல் பட அங்கீகாரம் மற்றும் பேச்சு செயலாக்கத்தில் புரட்சியை ஏற்படுத்தியுள்ளது.`
  - Reference: `Deep learning has revolutionized image recognition and speech processing.`
  - Hypothesis: `Than k ver y y y much kin help .`
#### te -> en
- **Sample 1**:
  - Source: `సహజ భాషా ప్రాసెసింగ్ టెక్స్ట్ యొక్క వాక్యనిర్మాణం మరియు అర్థాన్ని విశ్లేషిస్తుంది.`
  - Reference: `Natural language processing analyzes syntax and semantics of text.`
  - Hypothesis: `I am learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning learning`
- **Sample 2**:
  - Source: `డీప్ లెర్నింగ్ ఇమేజ్ రికగ్నిషన్ మరియు స్పీచ్ ప్రాసెసింగ్‌లో విప్లవాత్మక మార్పులు తెచ్చింది.`
  - Reference: `Deep learning has revolutionized image recognition and speech processing.`
  - Hypothesis: `The er generat es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es es`
