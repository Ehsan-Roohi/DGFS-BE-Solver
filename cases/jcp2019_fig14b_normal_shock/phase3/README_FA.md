# DGFS — مرحلهٔ ۳: ممیزی quadrature + projection محافظه‌کار پنج‌ممانی

این شاخه کد مرحلهٔ بعد است: (۳a) ممیزی کوتاه GPU روی snapshot زمان ۳۰ با سه quadrature و projection پنج‌ممانی. هوک restart مرحلهٔ ۳b آزمایشی است و فقط پس از پاس‌شدن گیت‌های GPU فاز ۳a باید استفاده شود. هیچ ران طولانی در فاز ۳a نیست.

## محتویات

```
p3/conservative_projection.py   projection پنج‌ممانی: مرجع numpy (euclidean, f, fplus, maxwellian)
                                + نسخهٔ GPU با pycuda (همان آدرس‌دهی AoSoA کرنل‌های vhs-gll)
p3/p3_analysis.py               تحلیل هر نقطه، خلاصهٔ هر quadrature، گیت‌ها، CSV
p3/audit_quadrature_projection.py   ممیزی GPU (اجرا روی Unity) — اصلی‌ترین فایل مرحلهٔ ۳a
p3/fs_reference.py              مرجع CPU اپراتور vhs-gll (با A100 تا ~1e-12 یکسان است)
p3/offline_quadrature_study.py  همان ممیزی بدون GPU (برای پیش‌بررسی؛ نتایجش در offline_results/)
p3/make_restart_configs.py      ساخت ۴ کانفیگ restart برای مرحلهٔ ۳b
p3/compare_restarts.py          مقایسهٔ snapshotهای نهایی ۴ ران (u_z، تنش، شار گرما، منفی‌بودن، H، زمان)
solver_hook/apply_hook.py       نصب هوک در checkout حل‌گر (کپی projection.py + وصلهٔ system.py)
solver_hook/system.py           نسخهٔ وصله‌شدهٔ مرجع system.py (دو بلوک اضافه شده: سازنده و collide)
solver_hook/projection.py       همان conservative_projection.py برای frfs/solvers/dgfs/
hpc/p3_quadrature_projection_audit.slurm   اسکریپت Slurm ممیزی ۳a
hpc/bootstrap_unity_p3_audit.sh            بوت‌استرپ یک‌خطی روی Unity (مثل مرحلهٔ ۲)
hpc/p3b_restarts.slurm                     چهار restart متوالی در یک job + مقایسه
offline_results/                نتایج پیش‌بررسی CPU روی همین snapshot (json/csv/log)
```

## اجرای مرحلهٔ ۳a روی Unity

```bash
curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DGFS-BE-Solver/agent/phase3-angular-conservative-audit/hpc/bootstrap_unity_p3_audit.sh | bash
```

متغیرهای اختیاری: `DGFS_P3_QUADRATURES=32:6,16:16,16:24` (ترتیب: baseline، proposed، reference)، `DGFS_P3_GPU_SOLVE=device|host`، `DGFS_P3_REPEATS`, `DGFS_P3_PROJ_REPEATS`.

خروجی: `p3_quadrature_projection_audit.json/.csv`، `p3_audit.log` (خطوط `P3_POINT`, `P3_CONTROL`, جدول خلاصه، فهرست `[PASS]/[FAIL]`) و ZIP در `$DGFS_ROOT/p3_audit_<jobid>.zip`.

حل ۵×۵ به‌صورت پیش‌فرض روی device انجام می‌شود و در هر نقطه با numpy مقایسه می‌شود (گیت G0). اگر G0 شکست خورد، با `DGFS_P3_GPU_SOLVE=host` تکرار کنید.

## اجرای مرحلهٔ ۳b (فقط اگر ۳a پاس شد)

```bash
mkdir p3b && cd p3b
cp <run>/dgfs_fig14b.ini <run>/mesh.frfsm <run>/dist_dgfs_fig14b-30.0.frfss <run>/kinetic_residual.csv .
cp -r <pkg>/p3 <pkg>/solver_hook . ; mkdir source && git clone --depth 1 --branch agent/phase2-collision-audit https://github.com/Ehsan-Roohi/DGFS-BE-Solver.git source/DGFS-BE-Solver
sbatch <pkg>/hpc/p3b_restarts.slurm
```

ران‌های پیشنهادی ۳b: `M6_raw:32:6:none, M6_fplus:32:6:fplus, M16_raw:16:16:none, M16_fplus:16:16:fplus`. وزن signed-`f` فقط یک تشخیص عددی است؛ چون snapshot دنباله‌های منفی دارد، نامزد تولیدی `fplus=max(f,0)` است. گزینهٔ کانفیگ: `[scattering-model] projection = none|euclidean|f|fplus` و `projection-solve = device|host`. با `projection = none` مسیر حل‌گر قبلی حفظ می‌شود.

## گیت‌ها (مقادیر پیش‌فرض، همه با سوییچ قابل تغییر)

| گیت | معیار | پیش‌فرض |
|---|---|---|
| G0 | تطابق Q_c روی GPU با numpy (rel L∞) | < 1e-10 |
| G1 | نقص پنج ناوردا پس از projection (cancellation-normalized) | < 1e-12 |
| G2 | نقص raw state-normalized با quadrature پیشنهادی | < 5e-5 |
| G3 | ‖δQ‖₂/‖Q‖₂ (جداگانه برای داخل شوک |x|<0.4 و همهٔ نقاط) | < 0.5% |
| G4 | نسبت جرم منفی (f+dt·Q_c)/(f+dt·Q) و تعداد گره‌های تازه‌منفی | ≤ 1.05 و ≤ 33 |
| G5 | min(f+dt·Q_c)/min(f+dt·Q) | ≤ 1.10 |
| G6 | کاهش منبع مصنوعی u_z (max|Q_z/ρ|) از baseline به proposed | ≥ 100× |
| G7 | زمان projection نسبت به collision | ≤ 5% |

## نتایج پیش‌بررسی آفلاین (CPU، همین snapshot، dt=0.001) — قبل از اجرای GPU بدانید

| quadrature | raw canc (همه) | raw canc (داخل) | raw state (همه) | max du_z/dt | هزینهٔ نسبی |
|---|---|---|---|---|---|
| Nrho32_M6 | 9.4e-3 | 5.1e-3 | 3.5e-3 | 4.0e-3 | 1.00 |
| Nrho16_M16 | 3.8e-3 | 2.1e-4 | 4.0e-5 | 1.3e-5 | ≈1.33 |
| Nrho16_M24 | 2.0e-3 | 1.8e-4 | 2.7e-5 | 1.6e-6 | ≈2.00 |

| quadrature | وزن | پس از projection | relL2 داخل | relL2 همه | نسبت جرم منفی | گرهٔ تازه‌منفی |
|---|---|---|---|---|---|---|
| M6 | euclidean | 4.5e-16 | 1.1e-3 | 1.1e-3 | **4.53** | **6502** |
| M6 | f | 5.1e-16 | **9.6e-3** | **1.8e-2** | 1.000 | 0 |
| M16 | euclidean | 3.7e-16 | 2.1e-5 | 9.4e-5 | 1.044 | **5848** (جرم ناچیز) |
| M16 | f | 3.6e-16 | 5.1e-4 | **1.1e-2** | 1.000 | 0 |
| M24 | euclidean | 4.0e-16 | 6.5e-6 | 9.0e-5 | 1.029 | 5524 |
| M24 | f | 3.1e-16 | 4.6e-4 | 6.1e-3 | 1.000 | 0 |

کنترل ماکسولی: ماکسولی ساکن با هر سه quadrature نقص تکانهٔ z در حد 1e-11 دارد؛ ماکسولی رانشی با حالت مرکز شوک (u_x≈1.27, T≈1.21) هم نقص z ≈ 4e-10 دارد ولی نقص تکانهٔ x آن 5.8e-4 (M6) → 3.3e-4 (M16) → 1.5e-4 (M24) است.

### تفسیر [قطعی از روی همین اعداد]
1. منبع مصنوعی u_z با M=16 حدود ۳۰۰ برابر و با M=24 حدود ۲۵۰۰ برابر کم می‌شود (G6 پاس). این منبع فقط برای f غیرتعادلی ظاهر می‌شود، نه برای ماکسولی رانشی؛ یعنی از برهم‌کنش ناهمسانگردی نیم‌کرهٔ icosahedral (یک رأس روی محور z + پنج نقطه با z=0.447) با تنش/شار گرمای شوک می‌آید.
2. نقص raw state-normalized با M=16 حدود 4e-5 است و با M=24 فقط به 2.7e-5 می‌رسد → کف باقی‌مانده از برش دامنهٔ سرعت/Nv=32 است، نه quadrature زاویه‌ای. گیت 1e-5 شما برای این شبکهٔ سرعت دست‌نیافتنی است؛ پیشنهاد: G2 را 5e-5 (state) بگذارید، یا Nv=48/L=8 را در یک سلول جداگانهٔ ماتریس حساسیت بیازمایید.
3. projection اقلیدسی با M=6 جرم منفی را ۲ تا ۴.۵ برابر می‌کند؛ با M=16 نسبت جرم منفی 1.04 است ولی هنوز ~۵۸۰۰ گرهٔ دنباله را به مقادیر منفی ناچیز می‌برد. در پیش‌بررسی، `fplus=max(f,0)` هیچ گرهٔ تازه‌منفی ایجاد نکرد؛ این یک مشاهدهٔ عددی برای همین snapshot و dt است، نه تضمین عمومی. بنابراین نامزد ۳b وزن `fplus` است.
4. relL2 وزن‌دار ذاتاً ۵ تا ۲۵ برابر اقلیدسی است (اقلیدسی به‌تعریف کمینهٔ L2 است) و در نقاط لبه (x=±0.5) که Q≈0 است، هر معیار نسبی به Q منفجر می‌شود؛ گیت 0.5% را روی |x|<0.4 ارزیابی کنید (با M=16 پاس می‌شود: 5e-4).
5. همهٔ نقص‌های پس از projection در حد 1e-16 هستند؛ روی GPU انتظار 1e-14 تا 1e-13 (G1 پاس).

## نکات پیاده‌سازی
- کرنل‌های projection: (۱) کاهش ۲۰ ممان (۵ ممان Q + ۱۵ درایهٔ گرام وزن‌دار) روی ۱۲۸ بلوک × ۲۵۶ نخ، (۲) کاهش نهایی + مقیاس قطری + حذف گاوسی با pivot در یک بلوک، (۳) اعمال Q ← Q − w·(Bᵀλ). همه double. هزینهٔ مورد انتظار ≪ 0.1 ms در هر نقطه.
- `fplus` = وزن max(f,0) (به‌لحاظ ریاضی تضمین عدم تولید منفی وقتی dt·max|Bᵀλ|<1)، `maxwellian` فقط در numpy (برای مقایسه).
- اگر تخصیص حافظهٔ GPU برای سه quadrature پشت‌سرهم مشکل‌ساز شد، هر quadrature را جداگانه با `DGFS_P3_QUADRATURES=16:16` اجرا کنید؛ فایل‌های `p3_partial_*.json` نوشته می‌شوند.
- کد CUDA روی این ماشین کامپایل نشده است (GPU در دسترس نبود)؛ گیت G0 دقیقاً برای گرفتن هر خطای احتمالی کرنل گذاشته شده است.
