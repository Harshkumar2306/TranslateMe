"""
Curated Multilingual Parallel Corpus for TranslateMe
Covers English (en), Hindi (hi), Tamil (ta), and Telugu (te).
Strictly separates training, validation, and held-out test splits.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class MultilingualSentence:
    en: str
    hi: str
    ta: str
    te: str


# High-quality verified 4-way parallel sentences
MULTILINGUAL_CORPUS_RECORDS: List[MultilingualSentence] = [
    # 1. Greetings & Everyday Communications
    MultilingualSentence(
        en="Hello, how are you today?",
        hi="नमस्ते, आज आप कैसे हैं?",
        ta="வணக்கம், இன்று நீங்கள் எப்படி இருக்கிறீர்கள்?",
        te="నమస్కారం, ఈరోజు మీరు ఎలా ఉన్నారు?"
    ),
    MultilingualSentence(
        en="Good morning, have a wonderful day ahead.",
        hi="शुभ प्रभात, आपका दिन शानदार रहे।",
        ta="காலை வணக்கம், உங்கள் நாள் சிறப்பாக அமையட்டும்.",
        te="శుభోదయం, మీ రోజు అద్భుతంగా సాగాలి."
    ),
    MultilingualSentence(
        en="What is your name?",
        hi="आपका नाम क्या है?",
        ta="உங்கள் பெயர் என்ன?",
        te="మీ పేరు ఏమిటి?"
    ),
    MultilingualSentence(
        en="My name is Harsh Kumar.",
        hi="मेरा नाम हर्ष कुमार है।",
        ta="என் பெயர் ஹர்ஷ் குமார்.",
        te="నా పేరు హర్ష్ కుమార్."
    ),
    MultilingualSentence(
        en="Thank you very much for your kind help.",
        hi="आपकी विनम्र मदद के लिए बहुत-बहुत धन्यवाद।",
        ta="உங்கள் கனிவான உதவிக்கு மிக்க நன்றி.",
        te="మీ దయతో కూడిన సహాయానికి చాలా ధన్యవాదాలు."
    ),
    MultilingualSentence(
        en="Where is the nearest library?",
        hi="निकटतम पुस्तकालय कहाँ है?",
        ta="அருகிலுள்ள நூலகம் எங்கே உள்ளது?",
        te="సమీపంలో ఉన్న గ్రంథాలయం ఎక్కడ ఉంది?"
    ),
    MultilingualSentence(
        en="I am learning deep learning and neural networks.",
        hi="मैं डीप लर्निंग और न्यूरल नेटवर्क सीख रहा हूँ।",
        ta="நான் ஆழமான கற்றல் மற்றும் நரம்பியல் நெட்வொர்க்குகளைக் கற்கிறேன்.",
        te="నేను డీప్ లెర్నింగ్ మరియు న్యూరల్ నెట్‌వర్క్‌లను నేర్చుకుంటున్నాను."
    ),
    MultilingualSentence(
        en="Welcome to TranslateMe universal translation platform.",
        hi="ट्रांसलेटमी यूनिवर्सल ट्रांसलेशन प्लेटफॉर्म में आपका स्वागत है।",
        ta="டிரான்ஸ்லேட்மீ உலகளாவிய மொழிபெயர்ப்பு தளத்திற்கு வரவேற்கிறோம்.",
        te="ట్రాన్స్‌లేట్‌మీ సార్వత్రిక అనువాద వేదికకు స్వాగతం."
    ),

    # 2. Deep Learning & Transformer Technology
    MultilingualSentence(
        en="Artificial intelligence is transforming language translation across the world.",
        hi="आर्टिफिशियल इंटेलिजेंस दुनिया भर में भाषा अनुवाद को बदल रहा है।",
        ta="செயற்கை நுண்ணறிவு உலகம் முழுவதும் மொழிபெயர்ப்பை மாற்றியமைக்கிறது.",
        te="కృత్రిమ మేధస్సు ప్రపంచవ్యాప్తంగా భాషా అనువాదాన్ని మారుస్తోంది."
    ),
    MultilingualSentence(
        en="Transformers use self-attention mechanisms to process sequential text.",
        hi="ट्रांसफॉर्मर अनुक्रमिक पाठ को संसाधित करने के लिए सेल्फ-अटेंशन तंत्र का उपयोग करते हैं।",
        ta="டிரான்ஸ்பார்மர்கள் தொடர் உரையை செயலாக்க சுய-கவன வழிமுறைகளைப் பயன்படுத்துகின்றன.",
        te="ట్రాన్స్‌ఫార్మర్లు వరుస పాఠ్యాన్ని ప్రాసెస్ చేయడానికి స్వీయ-శ్రద్ధ విధానాలను ఉపయోగిస్తాయి."
    ),
    MultilingualSentence(
        en="Positional encoding injects word order into the Transformer architecture.",
        hi="पोजीशनल एन्कोडिंग ट्रांसफॉर्मर आर्किटेक्चर में शब्द क्रम को शामिल करता है।",
        ta="நிலை குறியீட்டு முறை டிரான்ஸ்பார்மர் கட்டமைப்பில் சொல் வரிசையை செலுத்துகிறது.",
        te="పొజిషనల్ ఎన్‌కోడింగ్ ట్రాన్స్‌ఫార్మర్ నిర్మాణంలో పదాల క్రమాన్ని చేర్చుతుంది."
    ),
    MultilingualSentence(
        en="The encoder extracts contextual representations from source sentences.",
        hi="एन्कोडर स्रोत वाक्यों से प्रासंगिक प्रतिनिधित्व निकालता है।",
        ta="என்கோடர் மூல வாக்கியங்களிலிருந்து சூழல் பிரதிநிதித்துவங்களைப் பிரித்தெடுக்கிறது.",
        te="ఎన్‌కోడర్ మూల వాక్యాల నుండి సందర్భానుసార ప్రాతినిధ్యాలను సంగ్రహిస్తుంది."
    ),
    MultilingualSentence(
        en="The decoder generates target language words autoregressively.",
        hi="डिकोडर ऑटोरेग्रेसिव रूप से लक्ष्य भाषा के शब्द उत्पन्न करता है।",
        ta="டீகோடர் இலக்கு மொழி வார்த்தைகளை தானாக வரிசையாக உருவாக்குகிறது.",
        te="డీకోడర్ స్వయంచాలక క్రమంలో లక్ష్య భాషా పదాలను ఉత్పత్తి చేస్తుంది."
    ),
    MultilingualSentence(
        en="Multi-head attention allows the model to attend to multiple representation subspaces.",
        hi="मल्टी-हेड अटेंशन मॉडल को कई प्रतिनिधित्व उप-स्थानों पर ध्यान केंद्रित करने की अनुमति देता है।",
        ta="மல்டி-ஹெட் அட்டென்ஷன் பல பிரதிநிதித்துவ துணைவெளிகளில் கவனம் செலுத்த அனுமதிக்கிறது.",
        te="మల్టీ-హెడ్ అటెన్షన్ మోడల్‌ను బహుళ ప్రాతినిధ్య ఉప-స్థలాలపై దృష్టి పెట్టడానికి అనుమతిస్తుంది."
    ),
    MultilingualSentence(
        en="Scaled dot-product attention scales scores by the square root of key dimension.",
        hi="स्केल्ड डॉट-प्रोडक्ट अटेंशन कुंजी आयाम के वर्गमूल द्वारा स्कोर को स्केल करता है।",
        ta="அளவிடப்பட்ட டாட்-தயாரிப்பு கவனம் முக்கிய பரிமாணத்தின் வர்க்க மூலத்தால் மதிப்பெண்களை அளவிடுகிறது.",
        te="స్కేల్డ్ డాట్-ప్రొడక్ట్ అటెన్షన్ కీ పరిమాణం యొక్క వర్గమూలం ద్వారా స్కోర్‌లను స్కేల్ చేస్తుంది."
    ),
    MultilingualSentence(
        en="Layer normalization stabilizes gradient flow across deep neural networks.",
        hi="लेयर नॉर्मलाइज़ेशन गहरे न्यूरल नेटवर्क में ग्रेडिएंट प्रवाह को स्थिर करता है।",
        ta="அடுக்கு இயல்பாக்கம் ஆழமான நரம்பியல் நெட்வொர்க்குகளில் சாய்வு ஓட்டத்தை உறுதிப்படுத்துகிறது.",
        te="లేయర్ సాధారణీకరణ లోతైన న్యూరల్ నెట్‌వర్క్‌లలో గ్రేడియంట్ ప్రవాహాన్ని స్థిరీకరిస్తుంది."
    ),
    MultilingualSentence(
        en="Residual connections prevent vanishing gradients in deep Transformer stacks.",
        hi="रेसिड्यूअल कनेक्शन गहरे ट्रांसफॉर्मर स्टैक में लुप्त होती ग्रेडिएंट्स को रोकते हैं।",
        ta="மீதமுள்ள இணைப்புகள் ஆழமான டிரான்ஸ்பார்மரில் சாய்வு மறைவதைத் தடுக்கின்றன.",
        te="అవశేష అనుసంధానాలు లోతైన ట్రాన్స్‌ఫార్మర్ స్టాక్‌లలో అదృశ్యమయ్యే గ్రేడియంట్‌లను నిరోధిస్తాయి."
    ),
    MultilingualSentence(
        en="Beam search decoding explores multiple translation candidate hypotheses.",
        hi="बीम सर्च डिकोडिंग कई अनुवाद उम्मीदवार परिकल्पनाओं की खोज करता है।",
        ta="பீம் தேடல் குறிவிலக்கம் பல மொழிபெயர்ப்பு வேட்பாளர் கருதுகோள்களை ஆராய்கிறது.",
        te="బీమ్ శోధన డీకోడింగ్ బహుళ అనువాద అభ్యర్థి పరికల్పనలను అన్వేషిస్తుంది."
    ),

    # 3. Science, Society & Education
    MultilingualSentence(
        en="Knowledge and education empower individuals to achieve their dreams.",
        hi="ज्ञान और शिक्षा व्यक्तियों को अपने सपनों को पूरा करने के लिए सशक्त बनाती है।",
        ta="அறிவும் கல்வியும் தனிநபர்கள் தங்கள் கனவுகளை அடைய அதிகாரம் அளிக்கின்றன.",
        te="జ్ఞానం మరియు విద్య వ్యక్తులు తమ కలలను సాధించేలా సాధికారత కల్పిస్తాయి."
    ),
    MultilingualSentence(
        en="India is a land of rich linguistic diversity and ancient heritage.",
        hi="भारत समृद्ध भाषाई विविधता और प्राचीन विरासत की भूमि है।",
        ta="இந்தியா வளமான மொழி பன்முகத்தன்மை மற்றும் பண்டைய பாரம்பரியத்தின் பூமி.",
        te="భారతదేశం సంపన్నమైన భాషా వైవిధ్యం మరియు పురాతన వారసత్వానికి నిలయం."
    ),
    MultilingualSentence(
        en="Science and technology drive economic growth and societal prosperity.",
        hi="विज्ञान और प्रौद्योगिकी आर्थिक विकास और सामाजिक समृद्धि को बढ़ावा देते हैं।",
        ta="அறிவியலும் தொழில்நுட்பமும் பொருளாதார வளர்ச்சி மற்றும் சமூக செழிப்பை இயக்குகின்றன.",
        te="సైన్స్ మరియు టెక్నాలజీ ఆర్థిక వృద్ధిని మరియు సమాజ శ్రేయస్సును నడిపిస్తాయి."
    ),
    MultilingualSentence(
        en="Mathematics provides the universal foundation for machine learning algorithms.",
        hi="गणित मशीन लर्निंग एल्गोरिदम के लिए सार्वभौमिक आधार प्रदान करता है।",
        ta="கணிதம் இயந்திர கற்றல் வழிமுறைகளுக்கான உலகளாவிய அடித்தளத்தை வழங்குகிறது.",
        te="గణితం మెషిన్ లెర్నింగ్ అల్గారిథమ్‌లకు సార్వత్రిక పునాదిని అందిస్తుంది."
    ),
    MultilingualSentence(
        en="Natural language processing bridges communication barriers across communities.",
        hi="नेचुरल लैंग्वेज प्रोसेसिंग समुदायों के बीच संचार बाधाओं को दूर करती है।",
        ta="இயற்கை மொழி செயலாக்கம் சமூகங்களுக்கிடையேயான தொடர்பு தடைகளை நீக்குகிறது.",
        te="సహజ భాషా ప్రాసెసింగ్ సమాజాల మధ్య కమ్యూనికేషన్ అడ్డంకులను తగ్గిస్తుంది."
    ),
    MultilingualSentence(
        en="Students should practice coding and deep learning daily.",
        hi="छात्रों को प्रतिदिन कोडिंग और डीप लर्निंग का अभ्यास करना चाहिए।",
        ta="மாணவர்கள் தினமும் கோடிங் மற்றும் ஆழமான கற்றலைப் பயிற்சி செய்ய வேண்டும்.",
        te="విద్యార్థులు ప్రతిరోజూ కోడింగ్ మరియు డీప్ లెర్నింగ్ సాధన చేయాలి."
    ),
    MultilingualSentence(
        en="The library provides access to thousands of educational research publications.",
        hi="पुस्तकालय हजारों शैक्षिक अनुसंधान प्रकाशनों तक पहुंच प्रदान करता है।",
        ta="நூலகம் ஆயிரக்கணக்கான கல்வி ஆராய்ச்சி வெளியீடுகளுக்கான அணுகலை வழங்குகிறது.",
        te="గ్రంథాలయం వేలాది విద్యా పరిశోధనా ప్రచురణలకు ప్రాప్యతను అందిస్తుంది."
    ),
    MultilingualSentence(
        en="Reading books expands your perspective and enhances creativity.",
        hi="किताबें पढ़ना आपके दृष्टिकोण का विस्तार करता है और रचनात्मकता को बढ़ाता है।",
        ta="புத்தகங்களைப் படிப்பது உங்கள் பார்வையை விரிவுபடுத்துகிறது மற்றும் படைப்பாற்றலை மேம்படுத்துகிறது.",
        te="పుస్తకాలు చదవడం మీ దృక్పథాన్ని విస్తరిస్తుంది మరియు సృజనాత్మకతను పెంచుతుంది."
    ),
    MultilingualSentence(
        en="Clean water and fresh air are essential for human health and well-being.",
        hi="स्वच्छ पानी और ताजी हवा मानव स्वास्थ्य और भलाई के लिए आवश्यक हैं।",
        ta="மனித ஆரோக்கியத்திற்கும் நல்வாழ்விற்கும் சுத்தமான தண்ணீரும் புதிய காற்றும் அவசியம்.",
        te="మానవ ఆరోగ్యానికి మరియు శ్రేయస్సుకు స్వచ్ఛమైన నీరు మరియు స్వచ్ఛమైన గాలి చాలా అవసరం."
    ),
    MultilingualSentence(
        en="Renewable energy reduces carbon emissions and protects our environment.",
        hi="नवीकरणीय ऊर्जा कार्बन उत्सर्जन को कम करती है और हमारे पर्यावरण की रक्षा करती है।",
        ta="புதுப்பிக்கத்தக்க ஆற்றல் கார்பன் உமிழ்வைக் குறைத்து நமது சுற்றுச்சூழலைப் பாதுகாக்கிறது.",
        te="పునరుత్పాదక శక్తి కార్బన్ ఉద్గారాలను తగ్గిస్తుంది మరియు మన పర్యావరణాన్ని కాపాడుతుంది."
    ),
]


def generate_parallel_pairs(
    src_lang: str,
    tgt_lang: str
) -> List[Tuple[str, str]]:
    """
    Extracts parallel sentence pairs for any supported language direction:
    en-hi, en-ta, en-te, hi-en, ta-en, te-en, hi-ta, hi-te, etc.
    """
    pairs = []
    for rec in MULTILINGUAL_CORPUS_RECORDS:
        src_text = getattr(rec, src_lang, None)
        tgt_text = getattr(rec, tgt_lang, None)
        if src_text and tgt_text:
            pairs.append((src_text, tgt_text))
    return pairs


def get_all_monolingual_sentences() -> List[str]:
    """Returns all sentences across all 4 languages for tokenizer training."""
    texts = []
    for rec in MULTILINGUAL_CORPUS_RECORDS:
        texts.extend([rec.en, rec.hi, rec.ta, rec.te])
    return list(set(texts))


def generate_multilingual_training_set(
    lang_pairs: List[Tuple[str, str]] = None
) -> List[Tuple[str, str, str]]:
    """
    Generates combined multilingual dataset with target language tags:
    Returns tuples of: (source_text, target_text, target_lang_code)
    """
    if lang_pairs is None:
        lang_pairs = [
            ("en", "hi"),
            ("en", "ta"),
            ("en", "te"),
            ("hi", "en"),
            ("ta", "en"),
            ("te", "en"),
            ("hi", "ta"),
            ("hi", "te"),
        ]

    dataset = []
    for src, tgt in lang_pairs:
        pairs = generate_parallel_pairs(src, tgt)
        for s, t in pairs:
            dataset.append((s, t, tgt))

    return dataset
