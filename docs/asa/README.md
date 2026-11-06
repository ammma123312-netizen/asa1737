# ASA: تشغيل نماذج ضخمة (700 مليار معلمة) على عتاد محدود

> هذا المجلد هو جزء **ASA** من المستودع: أبحاث وأدوات لتشغيل نماذج مفتوحة
> المصدر عملاقة على أجهزة صغيرة أو على GitHub Actions، ثم معالجة البطء الناتج.

---

## 1. الجواب المختصر: هل يشغّل Colibrì نموذجاً بحجم 700 مليار معلمة على GitHub؟

**على المشغّلات المجانية: لا. وعلى مشغّل أكبر أو مشغّل ذاتي (self-hosted): نعم، لكنه بطيء.**

[Colibrì](https://github.com/JustVugg/colibri) محرّك مكتوب بلغة C. يشغّل نموذج GLM-5.2
(744 مليار معلمة، بنية MoE) بذاكرة RAM تقارب 25 جيجابايت. يبقي الأجزاء الكثيفة من
النموذج في الذاكرة، ويقرأ "الخبراء" (experts) من قرص NVMe عند الحاجة. لكن هذه
الطريقة توفّر **الذاكرة فقط، ولا توفّر القرص**: الأوزان نفسها حجمها 429 جيجابايت.

| | مشغّل GitHub المجاني (مستودع عام) | المطلوب لـ GLM-5.2 |
|---|---|---|
| المعالج | 4 أنوية | أي عدد |
| الذاكرة | 16 GB | 16–25 GB كحد أدنى |
| القرص | 14 GB مضمونة (≈70 GB بعد حذف الأدوات المثبتة مسبقاً) | **429 GB** |
| أقصى مدة للمهمة | 6 ساعات | تنزيل 429 GB وحده يستهلك جزءاً كبيراً منها |

لذلك ورشة `asa-colibri` تفحص الجدوى قبل التنزيل، وتوقف المهمة برسالة واضحة إذا كان
النموذج لا يتسع. أما النماذج التي **تعمل فعلاً** على المشغّل المجاني فهي
Qwen3-Coder-30B-A3B (19 GB) وQwen3.6-35B-A3B (23 GB) وOLMoE، وكلها من نوع MoE.

### ✅ تجربة فعلية: بنيت Colibrì وشغّلته بنفسي

جرّبته في حاوية مواصفاتها مطابقة تقريباً لمشغّل GitHub المجاني (4 أنوية، 15 GB RAM، 30 GB قرص)،
باستخدام Colibrì v2.1.0 (commit `0a5504a`) ونموذج Qwen3-Coder-30B-A3B:

| الخطوة | النتيجة |
|---|---|
| `./setup.sh` | 49 ثانية |
| `make qwen36` | 39 ثانية |
| تنزيل الأوزان (19.4 GB) | 73 ثانية |
| `coli doctor` | جاهز (مع تحذير: بعض الخبراء يُقرؤون من القرص، والإقامة المتوقعة في الذاكرة 23%) |
| `coli run --ngen 64` | جواب صحيح في 54 ثانية، تشمل تحميل النموذج |
| `coli serve` + `/v1/chat/completions` | يعمل؛ واجهة متوافقة مع OpenAI |
| سؤال بالعربية (30 رمزاً) | «العاصمة… هي الرياض» في 24 ثانية |
| السرعة الثابتة | **≈2.0 رمز/ثانية** (بين 1.97 و2.16) |
| ذاكرة المحرك | 3.4 GB فقط |

تقدير `asa.planner` لهذا الجهاز كان 1.18 رمز/ثانية، أي ضمن ضعفين من القياس. الاختبار
`test_estimate_matches_measured_run_in_4cpu_container` يثبّت هذه النتيجة.

**وحتى لو نجح التشغيل، يبقى بطيئاً.** هذا ما توقّعته في سؤالك، والأرقام تؤكده:

| الجهاز | سرعة GLM-5.2 المقيسة (حسب Colibrì) | تقدير `asa.planner` |
|---|---|---|
| حاسوب بذاكرة 64 GB وقرص NVMe | ≈0.8 رمز/ثانية | ≈0.5 رمز/ثانية |
| محطة عمل بذاكرة 128 GB | ≈1.83 رمز/ثانية | ≈1.0 رمز/ثانية |
| حاسوب بذاكرة 32 GB وقرص USB خارجي | ≈0.1 رمز/ثانية | — |

بهذه السرعة يستغرق جواب من 200 رمز ما بين **3 و10 دقائق**. لا يمكن تسريع النموذج
الضخم نفسه كثيراً على هذا العتاد، فالحل هو **ألّا يُستخدم النموذج الضخم إلا عند
الحاجة**، وأن يعمل في الخلفية. هذا موضوع القسم 3.

## 2. لماذا هو بطيء؟ (الحساب الذي يعتمد عليه المخطط)

كل رمز (token) يولّده النموذج يتطلب قراءة الأوزان النشطة كلها مرة واحدة:

- في النموذج **الكثيف** (dense)، تُقرأ كل الأوزان مع كل رمز. Llama 405B بتكميم 4 بت
  يعني نحو 243 GB لكل رمز.
- في نموذج **MoE**، تُقرأ أوزان الخبراء المختارين فقط. GLM-5.2 ينشّط نحو 40 مليار
  معلمة، أي ما يقارب 23 GB لكل رمز.

ما يوجد من هذه الأوزان في الذاكرة يُقرأ بسرعة 50–200 GB/ث، وما لا يتسع يُقرأ من القرص
بسرعة 0.3–6 GB/ث. لذلك السرعة ≈ (عرض النطاق) ÷ (البايتات المقروءة لكل رمز). على هذه
المعادلة يُبنى الملف `asa/planner.py`، وقد ضُبط على أرقام Colibrì المنشورة، فيأتي
تقديره ضمن ضعفين من القياس الفعلي.

```bash
python -m asa.planner --list                                    # العتاد والنماذج المعرّفة
python -m asa.planner --hardware gha-free-public --all-models   # ماذا يتسع على المشغّل المجاني؟
python -m asa.planner --hardware gha-free-public --disk-gb 70 --model qwen3-coder-30b-a3b
python -m asa.planner --hardware workstation-128gb --model glm-5.2 --json
```

## 3. البنية المقترحة لحل البطء

```
                    ┌──────────────── LiteLLM (:4000) ─────────────────┐
  واجهة التطبيق ──► │ ذاكرة مؤقتة ──► موجّه (RouteLLM) ──► بدائل احتياطية │
                    └──────┬──────────────────┬───────────────────┬────┘
                           ▼                  ▼                   ▼
                  asa-fast (ثوانٍ)    asa-cloud (ثوانٍ)    asa-big (دقائق)
                  Qwen2.5-3B محلي      Groq / Cerebras       Colibrì + GLM-5.2
                  llama.cpp :8001      مجاني بحدود          :8000 أو GitHub Actions
```

1. **طبقة سريعة**: نموذج صغير محلي (موجود أصلاً في المستودع) يعطي جواباً أولياً فوراً.
2. **طبقة سحابية مجانية**: نموذج 70B على Groq أو Cerebras للأسئلة الأصعب، وسرعته مئات الرموز في الثانية.
3. **طبقة النموذج الضخم**: Colibrì، للمهام التي تحتاج أعلى جودة (البحث المعمّق، المراجعات).
   تعمل **في الخلفية**:
   - محلياً عبر `coli serve` ثم `asa-big` في LiteLLM، بمهلة 30 دقيقة.
   - أو على GitHub: افتح Issue، واكتب السؤال في نصّها، وأضف الوسم `asa-ask`. تعمل
     الورشة وتنشر الجواب تعليقاً عند الانتهاء. البطء هنا لا يزعج أحداً، لأن لا أحد
     ينتظر أمام الشاشة.
4. **ذاكرة مؤقتة**: السؤال المكرر يُجاب من الذاكرة المؤقتة فوراً. استبدلها بـ GPTCache أو
   redis-semantic ليُجاب السؤال *المشابه* أيضاً.
5. **فك الترميز التخميني (Speculative decoding)**: يتوقع النموذج الصغير عدة رموز ويتحقق
   منها الكبير دفعة واحدة. GLM-5.2 في Colibrì يفعل ذلك داخلياً عبر رأس MTP (2–3 رموز لكل تمريرة).
6. **التوزيع**: وضع cluster في Colibrì، أو exo، أو distributed-llama، لتقسيم الخبراء
   على عدة أجهزة يقرأ كل منها من قرصه بالتوازي.

ملف الإعداد: `asa/router/litellm.yaml`. بعد تشغيله يكفي ضبط `LLM_HOSTS=localhost:4000`.

## 4. المشاريع: 29 مشروعاً مفتوح المصدر

القائمة نفسها موجودة بصيغة قابلة للقراءة آلياً في `asa/projects.yaml`، ويتحقق منها الاختبار.

### أ. تشغيل نموذج أكبر من الذاكرة

| # | المشروع | دوره في ASA |
|---|---|---|
| 1 | [Colibrì](https://github.com/JustVugg/colibri) | يقرأ خبراء MoE من القرص؛ 744B بذاكرة 25 GB؛ واجهة متوافقة مع OpenAI ووضع cluster |
| 2 | [AirLLM](https://github.com/lyogavin/airllm) | يحمّل النموذج الكثيف طبقة بعد طبقة؛ 70B على GPU بذاكرة 4 GB؛ بطيء جداً |
| 3 | [llama.cpp](https://github.com/ggml-org/llama.cpp) | المحرك المرجعي؛ mmap يترك نظام التشغيل يحمّل ما لا يتسع |
| 4 | [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | نسخة من llama.cpp بأنوية CPU أسرع وحيل خاصة بنماذج MoE الضخمة |
| 5 | [KTransformers](https://github.com/kvcache-ai/ktransformers) | الخبراء على CPU والانتباه على GPU؛ DeepSeek 671B على بطاقة 24 GB |
| 6 | [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) | العصبونات "الساخنة" على GPU و"الباردة" على CPU |
| 7 | [FlexLLMGen](https://github.com/FMInference/FlexLLMGen) | تفريغ الأوزان والـ KV بين GPU وCPU والقرص، لمعالجة الدفعات |
| 8 | [DeepSpeed ZeRO-Inference](https://github.com/deepspeedai/DeepSpeed) | يمرّر الأوزان من NVMe إلى GPU لنماذج أكبر بكثير من ذاكرة البطاقة |

### ب. تصغير الأوزان (التكميم)

| # | المشروع | دوره |
|---|---|---|
| 9 | [BitNet.cpp](https://github.com/microsoft/BitNet) | استدلال بـ 1.58 بت على CPU؛ أسرع وأصغر بكثير |
| 10 | [Unsloth](https://github.com/unslothai/unsloth) | تكميم ديناميكي يصغّر DeepSeek-R1 671B من 720 GB إلى ≈131 GB |
| 11 | [HQQ](https://github.com/mobiusml/hqq) | تكميم سريع بلا بيانات معايرة (1–8 بت) |
| 12 | [GPTQModel](https://github.com/ModelCloud/GPTQModel) | تكميم 4 بت بأنوية لـ CPU وGPU |
| 13 | [LLM Compressor](https://github.com/vllm-project/llm-compressor) | ينتج نسخاً بصيغ FP8/INT4/AWQ يشغّلها vLLM بأقصى سرعة |

### ج. محركات خدمة سريعة

| # | المشروع | دوره |
|---|---|---|
| 14 | [vLLM](https://github.com/vllm-project/vllm) | PagedAttention والتجميع المستمر؛ المعيار في الإنتاجية |
| 15 | [SGLang](https://github.com/sgl-project/sglang) | يعيد استخدام البادئات المتكررة؛ سريع جداً مع الوكلاء |
| 16 | [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | أسرع فك ترميز لمستخدم واحد على GPU |
| 17 | [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | محرك بلغة Rust فيه تكميم فوري وفك ترميز تخميني |
| 18 | [MLC LLM](https://github.com/mlc-ai/mlc-llm) | يترجم النماذج لأي GPU وللهاتف والمتصفح (WebGPU) |

### د. توزيع نموذج واحد على عدة أجهزة

| # | المشروع | دوره |
|---|---|---|
| 19 | [Petals](https://github.com/bigscience-workshop/petals) | كل متطوع يخدم بضع طبقات من النموذج، بأسلوب BitTorrent |
| 20 | [exo](https://github.com/exo-explore/exo) | يجمع الهواتف والحواسيب على شبكة محلية في عنقود واحد |
| 21 | [distributed-llama](https://github.com/b4rtaz/distributed-llama) | توازي الموترات على CPU عبر Ethernet؛ أجهزة أكثر تعني سرعة أعلى |

### هـ. إخفاء البطء وتجنّبه

| # | المشروع | دوره |
|---|---|---|
| 22 | [EAGLE](https://github.com/SafeAILab/EAGLE) | فك ترميز تخميني؛ تسريع 2–3 أضعاف |
| 23 | [LMCache](https://github.com/LMCache/LMCache) | يعيد استخدام ذاكرة KV بين الطلبات |
| 24 | [GPTCache](https://github.com/zilliztech/GPTCache) | ذاكرة مؤقتة دلالية: السؤال المشابه يُجاب فوراً |
| 25 | [LiteLLM](https://github.com/BerriAI/litellm) | نقطة OpenAI واحدة أمام كل المحركات، مع بدائل احتياطية ومهل |
| 26 | [RouteLLM](https://github.com/lm-sys/RouteLLM) | موجّه متعلَّم: السؤال السهل للنموذج الصغير، والصعب للكبير |

### و. المنصات

| # | المشروع | دوره |
|---|---|---|
| 27 | [Ollama](https://github.com/ollama/ollama) | خادم نماذج محلي بأمر واحد، مناسب للطبقة السريعة |
| 28 | [LocalAI](https://github.com/mudler/LocalAI) | خادم متوافق مع OpenAI، فيه محركات كثيرة واتحاد P2P |
| 29 | [llamafile](https://github.com/Mozilla-Ocho/llamafile) | ملف تنفيذي واحد لكل نموذج؛ سهل النشر على مشغّل |

## 5. تقسيم العمل إلى مراحل

| المرحلة | المحتوى | الحالة |
|---|---|---|
| 1 | المخطط `asa/planner.py` + الكتالوج `asa/projects.yaml` + الاختبارات | ✅ تم |
| 2 | ورش GitHub Actions: `asa-ci` و`asa-colibri` و`asa-airllm` | ✅ تم (أوامر asa-colibri جُرّبت محلياً) |
| 3 | طبقة السرعة `asa/router/litellm.yaml` + هذا المستند | ✅ تم |
| 4 | بناء Colibrì وتشغيل Qwen3-Coder-30B فعلياً (4 أنوية / 15 GB)، وقياس السرعة، ومعايرة المخطط | ✅ تم (≈2 رمز/ثانية) |
| 4ب | تشغيل الورشة `asa-colibri` نفسها على GitHub | ⏭ التالي |
| 5 | مشغّل ذاتي (حاسوبك بقرص NVMe ≥ 500 GB) لتشغيل GLM-5.2 عبر Issues بالوسم `asa-ask` | ⏭ |
| 6 | ربط LiteLLM في `docker/entrypoint.sh` خلف متغير `ASA_ROUTER=1` | ⏭ |
| 7 | ذاكرة مؤقتة دلالية (GPTCache) وموجّه RouteLLM | ⏭ |
| 8 | تجربة وضع cluster في Colibrì على جهازين | ⏭ |

## 6. طريقة التشغيل

**على GitHub**: من تبويب *Actions* اختر **ASA Colibri run** ثم *Run workflow*. النموذج
الافتراضي `qwen3-coder-30b-a3b`. لتشغيل `glm-5.2` اكتب في خانة `runner` اسم مشغّل أكبر
أو `self-hosted`، واختر قيمة `hardware` المناسبة له.

**محلياً**:

```bash
git clone https://github.com/JustVugg/colibri && cd colibri/c
./setup.sh && make qwen36
./coli serve --model /nvme/qwen3-coder-30b --model-id colibri   # http://127.0.0.1:8000/v1
pip install 'litellm[proxy]'
litellm --config asa/router/litellm.yaml --port 4000
# في التطبيق: LLM_HOSTS=localhost:4000
```

## 7. حدود يجب معرفتها

- أرقام المخطط **تقديرات** تقارب الواقع ضمن ضعفين تقريباً، وليست قياسات.
- ذاكرة `actions/cache` في GitHub محدودة بـ 10 GB للمستودع، فتُنزَّل الأوزان من جديد في كل تشغيل.
- AirLLM لم يُجرَّب فعلياً بعد. وهو مصمم أساساً لبطاقات GPU. دعمه لـ CPU موجود منذ الإصدار v2.10.1، لكنه بطيء جداً.
- الوسم `asa-ask` لا يضيفه إلا من لديه صلاحية triage أو أعلى، فلا يستطيع غريب تشغيل الورشة على حسابك.

## المصادر

- [Colibrì على GitHub](https://github.com/JustVugg/colibri)
- [Better Stack: Colibri GLM-5.2 guide](https://betterstack.com/community/guides/ai/colibri-glm/)
- [GIGAZINE: Colibrì runs GLM-5.2 on 25 GB](https://gigazine.net/gsc_news/en/20260710-colibri-glm/)
- [AirLLM على GitHub](https://github.com/lyogavin/airllm)
