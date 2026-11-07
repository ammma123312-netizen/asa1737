## (أ) الإقلاع والتجهيز · Startup and staging

- تحميل النموذج عند المنسّق · engine `loaded in`: **345.67 s**

| الخطوة · step | المنسّق · coordinator (s) | العمال · workers min / median / max (s) |
|---|---|---|
| Set up job | 1.0 | 0.0 / 1.0 / 1.0 |
| Job clock (the task stops in time to upload its results) | 0.0 | – |
| Run actions/checkout@v4 | 1.0 | 1.0 / 1.0 / 2.0 |
| Run actions/setup-python@v5 | 1.0 | 0.0 / 0.0 / 1.0 |
| Run pip install -q "PyJWT[crypto]" | 2.0 | 1.0 / 2.0 / 4.0 |
| Run actions/download-artifact@v4 | 1.0 | – |
| Preflight (services) | 1.0 | 1.0 / 2.0 / 3.0 |
| Node setup (dense + draft head staged locally; engram via HFFS) | 205.0 | – |
| Wait for workers, authenticated tunnels, probe | 242.0 | – |
| Start the OpenAI endpoint (763B, native precision, distributed) | 350.0 | – |
| Public API gateway (only if the ASA_API_KEY secret exists) | 0.0 | – |
| Run the requested task | 6951.0 | – |
| Keep the gateway open for the session | 0.0 | – |
| Report | 0.0 | – |
| Answer the issue | 0.0 | – |
| Tell the issue the run failed | 0.0 | – |
| Run actions/upload-artifact@v4 | 1.0 | 0.0 / 1.0 / 2.0 |
| Release workers | 0.0 | – |
| Post Run actions/setup-python@v5 | 0.0 | 0.0 / 0.0 / 1.0 |
| Post Run actions/checkout@v4 | 1.0 | 0.0 / 0.0 / 1.0 |
| Complete job | 0.0 | 0.0 / 0.0 / 1.0 |
| Free disk for the expert shard | – | 50.0 / 81.0 / 175.0 |
| Node setup + stage this worker's experts in RAM (tmpfs), overflow on disk | – | 133.0 / 178.0 / 190.0 |
| Start expert worker behind the authenticated guard | – | 4.0 / 4.0 / 5.0 |
| Serve until the coordinator finishes | – | 7435.0 / 7502.0 / 7556.0 |
| Worker logs | – | 0.0 / 0.0 / 1.0 |

## (ب) الثواني لكل رمز · Seconds per token

| المقياس · metric | القيمة · value |
|---|---|
| أول رمز · first-token latency (s) | 202.1 |
| الفاصل الوسيط بين الرموز · median inter-token gap (s) | 16.8 |
| الفاصل p90 · p90 inter-token gap (s) | 17.8 |
| المتوسط الكلي · overall s/token | 17.4 |
| المصدر · source | tokens (400 tokens) |
| التوقف · stopped / finish_reason | None / length |

## (ج) الحساب مقابل الشبكة · Compute vs network

| العامل · worker | المسار · route | العنوان · address | حساب العامل · guard (s / calls) | انتظار المنسّق · connect (s / calls) | الشبكة · network (s) |
|---|---|---|---|---|---|
| 1 | relay | bore.pub:48205 | 272.8 / 4686 | 1078.9 / 4667 | 806.1 |
| 2 | relay | bore.pub:53241 | 229.2 / 3919 | 968.9 / 3895 | 739.7 |
| 3 | relay | bore.pub:65225 | 228.1 / 4537 | 918.5 / 4515 | 690.4 |
| 4 | relay | bore.pub:22672 | 259.1 / 3607 | 905.2 / 3584 | 646.1 |
| 5 | relay | bore.pub:37987 | 118.1 / 3910 | 903.7 / 3885 | 785.6 |
| 6 | relay | bore.pub:25480 | 292.2 / 4661 | 1151.8 / 4641 | 859.6 |
| 7 | relay | bore.pub:6242 | 157.3 / 4424 | 1035.4 / 4399 | 878.1 |
| 8 | relay | bore.pub:31655 | 207.2 / 4903 | 1025.9 / 4903 | 818.7 |
| 9 | relay | bore.pub:12912 | 51.7 / 4723 | 732.8 / 4681 | 681.1 |
| 10 | relay | bore.pub:33697 | 263.9 / 4537 | 1146.9 / 4514 | 883.0 |
| 11 | relay | bore.pub:7122 | 170.2 / 4393 | 1111.4 / 4372 | 941.2 |
| 12 | relay | bore.pub:58437 | 100.0 / 3864 | 935.7 / 3848 | 835.7 |
| 13 | relay | bore.pub:57420 | 32.2 / 4503 | 689.2 / 4471 | 657.0 |
| 14 | relay | bore.pub:13506 | 218.1 / 4037 | 959.4 / 4009 | 741.3 |
| 15 | relay | bore.pub:4300 | 93.7 / 4738 | 1069.6 / 4737 | 975.9 |
| 16 | relay | bore.pub:20026 | 184.7 / 5723 | 1173.2 / 5659 | 988.5 |
| 17 | relay | bore.pub:57227 | 213.1 / 6366 | 1240.9 / 6312 | 1027.8 |
| 18 | relay | bore.pub:23546 | 132.8 / 5140 | 1187.3 / 5114 | 1054.5 |
| 19 | relay | bore.pub:44791 | 30.4 / 4278 | 661.4 / 4277 | 631.0 |
| **مجموع الانتظار لكل العمال · connect total, all workers (19)** | | | | **18896.1** | |
| **المجموع للعمال ذوي القياسين · total, workers with both guard and connect (19)** | | | **3254.8** | **18896.1** | **15641.3** |

- أبطأ عامل · slowest worker: **17** (bore.pub:57227), connect **1240.9 s**, guard 213.1 s, network 1027.8 s. When the coordinator overlaps its requests to all workers (parallel-batch patch), each layer waits roughly for the slowest worker, not the sum.

متوسط لكل طبقة · per-layer average = total / (54 forwards × 40 layers), forwards from DSpark (written - accepted + 1 prefill):
- للعمال ذوي القياسين (19) · summed over the 19 paired workers: الحساب · compute **1506.9 ms**, الانتظار · wait **8748.2 ms**, الشبكة · network **7241.3 ms**
- انتظار متسلسل لكل العمال (19) · serial wait over all 19 workers: **8748.2 ms** per layer
- T(1) من سطر DSpark · from the DSpark gate line: **16.4 s** for one single-row forward
- مكافئ التمريرات · engine wall 201.4 s / T(1) = **12.3** single-row passes (a batched verify pass costs more than T(1), so this can exceed the forward count)

> هذا متوسط محسوب من المجاميع، وليس قياساً لكل طبقة. This is an average derived from totals, not a per-layer measurement.

## (د) الاستقرار · Stability

| البند · item | العدد · count |
|---|---|
| مهلات · timeouts | 9 |
| أنابيب أُغلقت مبكراً ثم أعيد الاتصال · pipes closed then reconnected | 0 |
| أخطاء أنابيب · pipe errors | 0 |
| ذاكرة/قرص · OOM / EDQUOT / ENOSPC / Killed | 0 |
| عمال لم يتصلوا · workers never connected | 0 of 19 |

- timeouts: `worker-logs-15/guard.log`: `[guard] rejected ('127.0.0.1', 50106): TimeoutError`

- timeouts: `worker-logs-15/guard.log`: `[guard] rejected ('127.0.0.1', 50128): TimeoutError`

- timeouts: `worker-logs-15/guard.log`: `[guard] rejected ('127.0.0.1', 50138): TimeoutError`

- timeouts: `worker-logs-15/guard.log`: `[guard] rejected ('127.0.0.1', 50146): TimeoutError`

- timeouts: `worker-logs-15/guard.log`: `[guard] rejected ('127.0.0.1', 37890): TimeoutError`
