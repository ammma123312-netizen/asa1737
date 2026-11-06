# ASA: حاسوبك كمشغّل ذاتي (المرحلة 5)

يضيف السكربت `scripts/asa/self_hosted_runner.sh` حاسوبك (Linux) إلى سرب ASA كمشغّل GitHub Actions ذاتي
بالوسوم `self-hosted, asa`. لا يدفع أحد شيئاً: GitHub لا يحتسب دقائق المشغّلات الذاتية.

## المتطلبات
- Linux على x86_64 أو aarch64، ومعه `curl` و`tar` و`sha256sum`.
- يُفضّل 60 GB فارغة على قرص NVMe، لأن مهام السرب تنزّل أوزان الخبراء. السكربت يحذّرك إن كانت المساحة أقل.
- حساب مستخدم عادي، لا root. والأفضل حساب مخصص للمشغّل وحده.

## التثبيت
1. في GitHub افتح المستودع، ثم **Settings → Actions → Runners → New self-hosted runner**، وانسخ رمز التسجيل.
   الرمز صالح لساعة واحدة ولاستخدام واحد.
2. شغّل:
   ```bash
   scripts/asa/self_hosted_runner.sh install https://github.com/OWNER/REPO
   ```
   الصق الرمز عند الطلب. لن يظهر على الشاشة، ولن يُحفظ في ملف، ولن يمر في سطر الأوامر.
3. يعمل المشغّل كخدمة مستخدم (`asa-runner.service`). لتبقى تعمل بعد تسجيل الخروج:
   ```bash
   sudo loginctl enable-linger $(id -un)
   ```
4. تحقّق في صفحة Runners أن حالته **Idle**.

يتحقق السكربت من بصمة sha256 الرسمية للمشغّل قبل فك ضغطه، ويرفض المتابعة إن لم تتطابق.

## الحالة والإزالة
```bash
scripts/asa/self_hosted_runner.sh status
scripts/asa/self_hosted_runner.sh remove   # يطلب رمز إزالة من نفس الصفحة: Runners → اسم المشغّل → Remove
rm -rf ~/actions-runner-asa                # حذف الملفات بعد الإزالة
```

## إعدادات اختيارية
| المتغير | الافتراضي | الوظيفة |
|---|---|---|
| `ASA_RUNNER_DIR` | `~/actions-runner-asa` | مجلد التثبيت |
| `ASA_RUNNER_NAME` | `asa-<اسم الجهاز>` | اسم المشغّل في GitHub |
| `ASA_RUNNER_LABELS` | `asa` | وسوم إضافية |
| `ASA_RUNNER_VERSION` | أحدث إصدار | إصدار محدد من المشغّل |
| `ASA_RUNNER_DRY_RUN` | `0` | `1` يطبع الأوامر دون تنفيذها |

## تحذير أمني
المشغّل الذاتي ينفّذ كود الورش على جهازك بصلاحيات المستخدم الذي ثبّته.
- **لا تربطه بمستودع عام** إلا إذا كانت ورش طلبات الدمج من النسخ (forks) تتطلب موافقتك
  (Settings → Actions → General → Fork pull request workflows).
- استخدم حساب مستخدم مخصصاً ليس فيه ملفاتك الشخصية أو مفاتيح SSH.
- أزل المشغّل حين لا تحتاجه.

## الربط بالسرب
ورشة `asa-swarm.yml` تستهدف الوسم `asa` عبر مدخل `runner` يضيفه مالكها. حتى يُضاف، يظهر المشغّل في
GitHub لكنه لا يستلم مهام السرب.
