# Data prerequisites — WWI placebo populations and HMD revision policy

Closes `docs/STATUS.md` item 11. Written 2026-08-26 against the HMD release pinned in
`PREREGISTRATION.md` (bulk files last modified 2026-06-15 = current release,
DOI `10.4054/HMD.Countries.20260615`, Methods Protocol v6 rev. 2025-08-05).

Sources: public mortality.org country pages, the per-country *Background and
Documentation* PDFs (`…/hmd.v6/<CC>/Public/InputDB/<CC>com.pdf`), Methods
Protocol v6, MP Summary, FAQ (`/Project/FAQ`), What's New, Citation Guidelines,
Zipped Data Files page, STMF metadata. **Not consulted:** the per-country
`<CC>note.pdf` "Notes" files — they sit behind the login wall (the URL returns the
login HTML), and no login was attempted. Quotes below are verbatim from the
extracted PDF text; page/section pointers are given where the PDF has them.

---

## A. WWI-era data quality — placebo regime (train ≤1913, test 1914–1922)

### A.0 Two HMD-wide facts that apply to every placebo population

1. **1918 flu handling in HMD is a Lexis-triangle splitting effect, not a death-count
   adjustment.** Methods Protocol v6, Appendix A ("Linear model for splitting 1×1
   death counts"): *"The Spanish flu epidemic during the winter of 1918–1919 had the
   effect of increasing the proportions of lower-triangle deaths in 1918 (which
   includes more deaths from the second half of the calendar year) and of
   upper-triangle deaths in 1919 (for the opposite reason). Since this was a global
   epidemic, it seems reasonable to extrapolate the experience of Sweden and France
   (Japanese data begin later) onto the rest of the populations in the HMD."*
   Consequence for us: `Deaths_1x1` in 1918–19 is unaffected wherever the raw input
   was already a 1×1 square (the two triangles sum back to the raw count); only
   `Exposures` inherit a small, model-based 1918/19 correction. Negligible for
   age×year Poisson scoring; worth one sentence in the data appendix.
2. **No HMD population carries an explicit "1914–1922 quality" flag.** The flags that
   exist are about (i) civilian-vs-total coverage during 1912/14–1920 (GBR, FRA, ITA),
   (ii) territorial changes (FRA 1914/1920, DNK 1921, ITA 1924), and (iii) raw-data
   *format* changes that happen to fall in the window (DNK 1916/1921, FIN 1917,
   ITA 1915/1921, NLD 1918 infant definition, SWE 1918 source change).

### A.1 Per-population table

| Pop | Series type | Territory 1900–25 | Military deaths 1914–20 | 1918 flu | HMD-flagged concerns touching 1914–22 | Class |
|---|---|---|---|---|---|---|
| **CHE** | total, neutral | none | n/a | listed as a "specific episode" | none after 1900 | **CLEAN** |
| **DNK** | total, neutral | **South Jutland added 1921** (+163k, ≈5.5%) | n/a | not mentioned | deaths format: 5-yr groups →1916 single age →1921 by cohort | **CAVEAT** (minor) |
| **FIN** | total; Russian Grand Duchy →1917 | 1920 Petsamo in / Repola–Porajärvi out (tiny) | none (no conscription) | not mentioned | none; pop. estimates 1877–1939 only in 5-yr age groups | **CLEAN** (see note on 1918 civil war) |
| **FRATNP** | **total incl. military** (Vallin–Meslé reconstructions) | **1914: minus Alsace-Lorraine + 10 invaded départements; 1920: current territory** | **included, estimated** (males 1914–19) | listed as a "specific episode" | 1916 census cancelled; deaths of unknown age 1914–33; civilian variant FRACNP has HMD-flagged implausible 1919 denominators | **CAVEAT** |
| **GBRTENW** | **total incl. military abroad** (Jdanov et al. 2005) | none | **included, estimated** (males 1914–20) | listed as a "specific episode" | 1912–20 pop. and 1914–20 male deaths are reconstructions (RefCodes 98/99); age heaping during WWI | **CAVEAT** |
| **GBR_SCO** | **civilian only** (no total variant exists) | none | **excluded** (deaths abroad); pop. excludes military 1912–20 | not mentioned | sudden male pop. decline + demobilisation jump; 1912–38 pop. only in 5-yr groups, split by ad-hoc spline; cohort tables "questionable" | **CAVEAT** (strong; sensitivity-drop) |
| **ISL** | total, neutral, register-based | none | n/a | not mentioned | none (pre-1900 80+ only); n≈90k → Poisson noise dominates | **CLEAN** |
| **ITA** | **total incl. military** (Jdanov–Glei–Jasilionis 2008) | 1924 additions are **outside** the window; 1921 census adjusted to exclude them | **≈650k added back**, age and year distribution *estimated* | mentioned only via 1917–19 birth-cohort dents | pre-1906 "extra caution"; 1908 & 1915 earthquakes → many unknown-age deaths; raw deaths in 5-yr groups 1906–14 and 1921–25, triangles 1915–20 | **CAVEAT** |
| **NLD** | total, neutral (mobilised) | none | n/a | not mentioned | **infant-death definition changes 1918** (false stillbirths), CBS/NIDI-adjusted back to 1850; NIDI pop. estimates: "under-registration of sailors and military personnel … still have to be solved" | **CAVEAT** (minor; age 0) |
| **NOR** | total, neutral | none | n/a (seamen abroad: handling after 1910 undocumented) | not mentioned | none for the window; deaths from SSB internal DB since 1911 | **CLEAN** |
| **SWE** | total, neutral | none | n/a | listed, ages 15–40 | source switch (unpublished → published tables) at 1918, same triangle format | **CLEAN** |
| BEL (already excluded) | — | — | — | — | *"Warning: There is a gap in the HMD data series for years 1914-1918 in Belgium due to the lack of mortality statistics during this period"* | **EXCLUDE** (confirms prereg) |

No population upgrades to EXCLUDE. Recommended kept set: all 11, with the
sub-grouping in §C.

### A.2 Per-population evidence (quoted)

**CHE** — `CHEcom.pdf` (rev. 2025-10-21). Territory: *"To the best of the authors'
knowledge, there have been no territorial changes in Switzerland during the period
covered by the HMD data."* Flu: *"The influenza epidemics in 1918-19, 1926-27, and
1969-70 resulted in an increase in deaths and a decrease in births nine months
later."* Quality: *"There were some problems with the completeness of death
registration prior to 1900 … Other than these problems, the data are believed to be
of high quality."* Nothing on 1914–22. → CLEAN.

**DNK** — `DNKcom.pdf` (rev. 2026-04-10). Neutrality: *"Denmark stayed neutral during
World War I and managed to regain North Schleswig after the war."* Territory:
*"Until 1921, data on population and deaths refer to the territory of Denmark
excluding South Jutland … In 1920, South Jutland became part of Denmark again, and
population data for this area are incorporated in the database starting with the
following year. As a result, the mortality database covers two periods."* Format:
*"For the years prior to 1916, data on deaths are available only by five-year age
groups; for 1916-1920 by single year of age; and for 1921 onwards by single year of
age and by single year of birth. Data for 1916 and later are therefore of superior
quality than those for earlier periods. There is no convincing evidence for age
misreporting or age heaping."* → CAVEAT (minor): a +5.5 % territorial jump at
h=8 (1921) inside the test window. Rates are territory-invariant to first order and
HMD carries the jump in `Population` (the "1921−/1921+" convention), so **no
adjustment to rates**; just do not score *counts* or e₀ across 1920→1921 as if the
population were closed, and note that single-age deaths ≤1915 are HMD-split.

**FIN** — `FINcom.pdf` (rev. 2025-06-02). Territory: *"Prior to Finland independence in
1917, the country was under, first, Swedish (until 1809) and, then, Russian
(1809-1917) rule. In 1920, a treaty was signed between Russia and Finland under
which the northern area of Petsamo was incorporated into Finland while the
previously Finnish districts of Repola and Porajärvi were ceded to Russia."*
Format: deaths *"1878–1916 … (Lexis rectangles)"*, *"1917–1980 … (Lexis
triangles)"*; population *"1877–1939 Population estimates (as of December 31st)
0, 5…90+"* — i.e. single-age exposures in **both** train and test are HMD-split
from 5-year groups (not break-specific). No war, flu, or 1914–22 remark at all.
*Not in HMD docs, external fact for the paper text:* Finland's 1918 civil war (Jan–May)
produced ≈36k excess deaths, overwhelmingly young men, and these are domestic
civil-register deaths, not military-abroad deaths — so FIN's 1918 spike is a genuine
observed shock (civil war + flu), not a reconstruction. → CLEAN.

**FRATNP** — country page: *"The data given here cover the total population including
both civilian and military regardless of whether the death occurred abroad."* and
*"The period data in the two series differ only for 1914-1920 and 1940-45."*
`FRATNPcom.pdf` (rev. 2025-07-14): *"For periods during World War I and II, we use
death counts that have been adjusted by Vallin and Meslé (2001) to account for
military losses."* Territory table: *"1869-1913: Current territory minus
Alsace-Lorraine … 1914-1919: Current territory minus the areas affected by the
military operations (Alsace-Lorraine, plus Aisne, Ardennes, Marne,
Meurthe-et-Moselle, Meuse, Nord, Oise, Pas-de-Calais, Sommes, and Vosges) …
1920-1938: Current territory excluding a few areas within the Alpes-Maritimes"*;
*"we have not adjusted the data in an attempt to make it correspond to a constant
territory. Territorial changes are reflected in data files on population size, which
contain estimates … immediately before and after a territorial change ('1914–' /
'1914+')."* Census: *"The 1916 and 1941 censuses were cancelled due to World Wars
I and II."* Population: *"The data cover the entire national (de jure) population
(including military). Vallin and Meslé (2001) made adjustments to the population
estimates during World War I and II for military personnel."* Deaths appendix:
*"1907-1933 … Lexis triangles … unk (1907 and 1914-33 only)"* — deaths of unknown
age exist throughout the test window. Why **not** to swap in the civilian variant
FRACNP as a sensitivity — `FRACNPcom.pdf`: *"among males aged 20-21 in 1919,
civilian death rates are higher than the total population (e.g., M21=0.076 for
civilian males vs. M21=0.015 for all males … The main source of this difference
appears to be especially low civilian population estimates."* → CAVEAT: keep
FRATNP; the two territorial breaks (1914 = train/test boundary, 1920 = h 7) are
absorbed by exposures; male deaths at conscription ages 1914–19 are Vallin's
reconstructions, so age-resolved coverage for FRATNP males 18–45 in 1914–19
compares a forecast against an *estimate*, not an observation.

**GBRTENW** — country page: *"The data given here cover the total population including
both civilian and military regardless of whether the death occurred abroad."* /
*"The period data in the two series differ only for the years between the pre-war
and post-war censuses (1912-1920 and 1939-1950)."* `GBRTENWcom.pdf` (rev.
2025-01-20): *"Population estimates and deaths counts for the period during the two
World Wars were provided by the General Register Office and include only
information about the civilian population. For the total population (including
military), estimates of population (1912-1920 and 1932-1950) and death counts
(1914-1920 and 1939-1950) come from Jdanov et al. (2005)."* *"Death counts during
the two World Wars include both civilian and military deaths (even if the death
occurred abroad)."* *"Single-year death counts show some signs of age heaping
effects during World War I and between the two World Wars."* *"There were no
territorial changes in England and Wales during the period of reference
(1841-2022)."* Appendix 1: males 1914–1920 = *"Annual number of deaths (including
military deaths abroad), by sex, age, and year of birth"* RefCode 98; population
1912–1920 = *"Estimates during the war time include military persons stationed
abroad."* RefCode 99. → CAVEAT: same status as FRATNP (total series, male
1914–20 surface reconstructed). Keep.

**GBR_SCO** — country page: *"The data given here represent only the civilian
population. During the periods between pre-war and post-war censuses (1912-20 and
1939-50), population estimates exclude the military while death counts exclude
military deaths that occurred abroad."* and *"we caution the user that they [cohort
life tables] are of questionable value for cohorts that experienced significant
military losses during war."* `GBR_SCOcom.pdf` (rev. 2025-01-29; *"We are still
working on the Background and Documentation file for Scotland"*): *"During World
War I and World War II, the male population exhibits a sudden decline because the
military is not included. There is a corresponding increase in the male population
during the first post-war years due to demobilization. Population estimates for the
period around World War I are available by five-year age groups only.
Unfortunately, the standard HMD method for splitting such data into single years of
age does not work well due to the irregular implicit migration pattern. … a modified
spline method was used to split population estimates data for the periods 1911-1921
…"* Appendix: *"1912-1938 Annual mid-year population estimates … 0-4, 5-9, …, 85+
… The data for 1912-20 exclude the military population."* Also: *"the quality of the
data for 1855-1875 is assumed to be lower than in later years"* (inside the
training window). → CAVEAT (strong). GBR_SCO is a *civilian* series with an
HMD-flagged, ad-hoc-splined male denominator for 1912–21: its "WWI shock" is home
front + flu only, and male age-specific rates 18–40 in 1914–21 are partly denominator
artefacts. Keep for females and ages ≥45; pre-register a drop-GBR_SCO sensitivity.
Note the prereg already excludes GBRCENW as an overlapping variant, which is the
*same* civilian construction for England & Wales — consistency argues for treating
GBR_SCO as the odd one out, not for re-admitting GBRCENW.

**ISL** — `ISLcom.pdf` (rev. 2026-05-04): *"There were no significant territorial changes
in Iceland during the period covered by the Human Mortality Database."*; *"In 1918,
Iceland was declared a sovereign state in union with Denmark"*; population estimates
*"are considered to be of very good quality."* Country page warns only about
*"ages 80+ for years prior to 1900"*. Nothing on war or the 1918 flu (which did hit
Reykjavík — external fact). → CLEAN. Statistical, not documentary, caveat: ≈90k
population, so 1×1 cells are dominated by Poisson noise; ISL will stress the
count-scale (Poisson log score) more than any other placebo population.

**ITA** — `ITAcom.pdf` (rev. 2026-03-02): *"It is known, however, that the death counts
recorded in vital statistics exclude large numbers of military deaths during the two
World Wars (approximately 650,000 male deaths during World War I and 290,000 during
World War II). Therefore, the death counts have been adjusted to include military
deaths during the two wars (see Appendix II). Because these death counts are not
available by age and calendar year, they have been redistributed using special
methods (for details, see Jdanov et al., 2008)."* Population: *"For 1912-20 and
1937-51, the pre- and post-war census counts and deaths data for the total
population (including the military) are used to derive annual (January 1st)
population estimates. The intercensal survival method … except that no migration is
assumed (except for the mobilization and demobilization of military troops during
1915-18 and 1940-45 …)."* Territory: *"With the Paris peace settlement of World War I
(1919), the Venezia Tridentina region … was added … All of these territories were
included in the 1921 census, but they were not included in the vital statistics data
until 1924. Therefore, the 1921 census counts were adjusted to exclude these
territories"* — the change is at 1924, outside the window. Quality: *"The data prior
to 1906 should be used with extra caution due to problems of quality … Users are
advised to use the life tables by five-year (or 10-year) age groups"*; *"There were a
lot of deaths of unknown age in 1908 and early 1909 due to an earthquake on Dec 28th,
1908"*; *"There were also a lot of deaths of unknown age in 1915 due to an earthquake
on Jan 13th, 1915."* Appendix II: military WWI deaths *"Distribution by calendar year
was estimated (see NoteCode=30)"*, *"Distribution by age group was estimated (see
NoteCode=32)"* from a *"Sample of military deaths in the province of Bologna by age"*.
Raw format: 1906–14 single age to 14 then 5-year groups; 1915–20 Lexis triangles
(adjusted); 1921–25 5-year groups again. → CAVEAT. The male 18–41 surface for
1915–20 is an imputation (aggregate 650k × estimated year split × Bologna-sample age
split); the single-age surfaces on either side (≤1914, 1921–25) are HMD 5-year splits.
Keep, but (i) report ITA males at ages 18–45 in 1915–20 separately from the
head-line H4 age profile, (ii) prefer 5-year age aggregation for any ITA-specific
figure, (iii) decide explicitly whether the training window starts 1872 (HMD:
"extra caution" ≤1905) — the prereg's "start ≤1900" rule admits it; do not silently
change.

**NLD** — `NLDcom.pdf` (rev. 2025-09-17): *"There were no territorial changes in the
Netherlands during the period covered by the available data (1850-2023)."* Infant
definition: *"During the period 1850-1917, live-born children who died before
notification … were counted as stillbirths … Starting in 1918, these newborns were
classified as live births and infant deaths. … The CBS used such proportions to
adjust figures on live births and infant deaths for the period 1900-1923 to include
these 'false stillbirths.' The NIDI researchers completed this work, extending the
series … backwards to 1850."* Population: Tabeau et al. acknowledge *"problems
related to the definition of 'de jure population' and 'de facto population,'
under-registration of sailors and military personnel, and the definition of
still-born children still have to be solved. To date no attempt has been made to
introduce any adjustment"*. Nothing on 1914–18 mobilisation or the flu. → CAVEAT
(minor): age-0 series across 1918 is a harmonised reconstruction (uniform method on
both sides of the break, so no step, but not raw); exposures 1850–1949 are NIDI
estimates. Keep; mention for H4 at age 0.

**NOR** — `NORcom.pdf` (rev. 2025-09-11): *"To the best of the authors' knowledge all
population and vital statistics refer to the contemporary territory of Norway."*
*"Data on deaths since 1911 are available from an internal death database of Norway
Statistics (Statistisk Sentralbyrå, 1978)."* Seamen: for 1876–1910, *"The difference
between deaths registered among the resident population and all deaths registered
in Norway is mostly due to deaths among Norwegian seamen aged 15 to 60"* and *"Male
deaths at ages 15–59 have been adjusted later for deaths among Norwegian seamen
while abroad"*; nothing is said for 1911+, so treatment of the ≈2k Norwegian merchant
seamen lost 1914–18 (external fact) is undocumented. Births 1846–1915 not by sex
(split by a 1.06 ratio) → affects age-0 exposures only. Nothing on the flu. → CLEAN
(with the seamen line as a footnote).

**SWE** — `SWEcom.pdf` (rev. 2025-05-14): *"The Spanish influenza epidemic of 1918-19
also resulted in increased death rates, especially among those aged 15 to 40
(Hofsten and Lundström, 1976: p. 50). Because Sweden remained neutral during both
World Wars, its population was minimally affected relative to other European
countries."* *"There have been no significant territorial changes in Sweden during
the period covered by the Human Mortality Database."* *"Since 1860, population data
can be considered of very high quality."* Appendix: 1901–1954 deaths *"Data before
1918 come from unpublished sources, while later data come from published sources"*
(same Lexis-triangle format either side). → CLEAN. SWE is also one of the two
populations whose 1918/19 triangle proportions calibrate the HMD-wide flu
adjustment (A.0), so it is the least model-touched placebo series.

### A.3 What the placebo actually tests, per sub-group

- **Neutral / no reconstruction** (CHE, DNK, FIN, ISL, NLD, NOR, SWE): the 1914–22
  break is essentially the **1918–19 pandemic** (plus Finland's civil war). This is
  the closest historical analogue to 2020–24 — a pandemic hitting a stable trend —
  and none of its observed deaths are HMD imputations.
- **Belligerent, total series** (FRATNP, GBRTENW, ITA): pandemic **plus** war mortality
  of young men, where the war component is reconstructed by HMD-affiliated authors
  (Vallin–Meslé; Jdanov et al. 2005; Jdanov et al. 2008). Coverage failures at male
  ages 18–45 in 1914–20 are partly forecast-vs-estimate comparisons.
- **Belligerent, civilian series** (GBR_SCO): pandemic plus home-front, with an
  HMD-flagged denominator problem for men.

---

## B. Is the latest year (2024) final, or revised in later vintages?

**Short answer.** HMD publishes no per-year provisional flag and no revision
guarantee; *every* update re-releases the whole series ("revised and updated
through YYYY"), and the Methods Protocol explicitly labels post-censal population
estimates provisional. The death counts for 2024 are final at source for most of
the 20 shift populations, but the **exposures for every post-census year
(2021/22–2024) in census-based populations are provisional by construction**, and
two populations (USA, CHL) carry explicit 2024-specific caveats. So "2024 may move"
is true, but the mechanism is mostly denominator revision after the next census,
not late death registrations — and it affects 2021–2024, not 2024 alone.

### B.1 Policy statements found

- **Methods Protocol v6 §5.2.3 (Pre- and postcensal survival method), p. 23:** *"the
  formulas … lack a correction for migration/error. Therefore, population estimates
  for recent years that are derived in this manner must be considered provisional.
  They will be replaced by final estimates once another census is available to close
  the intercensal interval."* This is the only formal "provisional" statement in the
  protocol; there is none about death counts.
- **FAQ "How often is the HMD updated?":** *"The HMD is updated on a continuous basis
  as new data become available for each country."* No FAQ entry addresses
  revisions, versions, or provisional data.
- **Explanatory Notes page, Data Availability page, Zipped Data Files page:** no
  provisional/revision language. Zipped Data Files shows *"Current Version:
  06/15/2026 … DOI:10.4054/HMD.Countries.20260615"* and archives previous versions.
- **Citation Guidelines:** *"Available at www.mortality.org (data downloaded on
  [date], DOI: 10.4054/HMD.Countries.[version code])."* and *"The DOI of the current
  version can be found on the 'Zipped Data Files' page."* — i.e. HMD's own answer to
  vintage drift is "cite the DOI". Release DOIs exist since the 2025-09-29 update.
- **What's New log (2025-01-01 → 2026-08-26), per shift population** — every entry is
  phrased "revised and updated", and the USA shows the two-pass pattern in the log
  itself:

  | Pop | Last update | Reaches | Pop | Last update | Reaches |
  |---|---|---|---|---|---|
  | BEL | 2025-11-25 | 2024 | LTU | 2025-08-05 | 2024 |
  | CHE | 2025-11-25 | 2024 | LUX | 2026-02-18 | 2024 |
  | CHL | 2026-02-10 | 2024 | LVA | 2025-09-25 | 2024 |
  | DNK | 2026-04-29 | **2025** | NOR | 2025-09-25 | 2024 |
  | EST | 2025-11-10 | 2024 | PRT | 2025-12-11 | 2024 |
  | FIN | 2025-06-11 | 2024 | SVK | 2025-12-11 | 2024 |
  | HKG | 2026-02-10 | 2024 | SWE | 2025-06-11 | 2024 |
  | HRV | 2026-04-07 | 2024 | TWN | 2026-02-18 | 2024 |
  | ISL | 2026-06-03 | 2024 | USA | 2026-02-18 **and** 2026-06-15 *"(with newly released data)"* | 2024 |
  | JPN | 2025-12-11 | 2024 | KOR | 2026-04-07 | 2024 |

  Placebo-only populations in this vintage: FRATNP, ITA, NLD reach 2023; GBRTENW,
  GBR_SCO reach 2022.

### B.2 Population-specific notes on whether 2024 is preliminary

| Pop | Population input for 2021–24 | Explicit statement | 2024 death counts at source (STMF metadata proxy, §3.8 "Delays") |
|---|---|---|---|
| **USA** | Census Bureau post-censal, *"a blend of the Vintage 2020 Census Bureau estimates, demographic analysis, and the preliminary data from the 2020 Decennial census"* | `USAcom.pdf` (rev. 2026-06-10): *"we typically conduct two updates each year: a first, preliminary, update using the publicly available mortality data … and, once the restricted mortality files have been released to our RDC account, a second update … The current update uses the restricted-use mortality data (deaths by Lexis triangle) for 2024."* Also: *"Postcensal population estimates are frequently revised … 'With each new release of annual estimates, the entire time series of estimates is revised for all years back to the last census.'"* | Pinned vintage = the second (restricted-data) pass → deaths final; exposures will be re-estimated with every Census vintage and replaced by intercensal estimates after the 2030 census |
| **CHL** | INE estimates **still based on the 2017 census** | `CHLcom.pdf` (rev. 2026-01-11): *"INE estimates based on the 2017 Census vintage. As of January 6, 2026 the revised estimates based on the 2024 Census have not yet been released."* and *"the INE population estimates are based on the 2017 census and need to be revised, accounting for the 2024 censal figures."* HMD notes the 2024 official estimates over-state younger ages *"(revisions forthcoming)"*. | STMF: *"2016-2023 data are final. 2024-2026 data are being collected; thus, they are preliminary"* |
| **TWN** | household-register based | none | STMF: *"Data from 2000 to 2023 are final and results for 2024 are provisional"* |
| **JPN** | official post-censal since the 2020 census; HMD re-derives intercensal after each census (*"population estimates by age and sex are not revised by the Statistics Bureau after a new census … HMD uses the methods described in the Methods Protocol to produce intercensal estimates"*) | 2025 census results will trigger re-estimation of 2021–24 exposures | n/a in STMF |
| **PRT** | *"As of December 2025, population estimates for 2021 onward are postcensal and based on the 2021 census."* | — | STMF: *"Data for 2020-2024 is final"* |
| **SVK** | *"the most recent official post-censal population estimates for 2022-2025 (based on the 2021 census)"* | — | STMF: *"2000-2023 data are final. 2024, 2025 & 2026 data are updated weekly"* |
| **HRV** | *"official postcensal estimates based on the 2021 census for the years since then"* | — | STMF: *"2001-2023w52 data are final. 2024, 2025 and 2026 data are preliminary"* |
| **LTU** | *"The most recent post-censal estimates for January 1st, 2022-2025 are produced from the Statistical Population registers."* | — | STMF: *"2000-2024 data are final"* |
| **EST, LVA, LUX, BEL** | 2021-census post-censal / register-based | BEL: *"The final data from the last population census for 2021 was released in March 2024."* | STMF: EST *"2000-2024 … final"*; LVA *"2000-2025 … final"*; LUX *"2000-2024 … final"*; BEL *"2000 to 2022 are final. Death counts for 2023, 2024, 2025 and 2026 are preliminary"* |
| **HKG** | by-census 2021; 85+ via survivor-ratio/extinct-cohort | `HKGcom.pdf` Jan-2025 revision reworked 85+ inputs | n/a in STMF |
| **KOR** | KOSTAT register/census | none | STMF entry stale (through 2023) |
| **CHE** | register; *"2022-2023 definitive resident population was available only up to age 105+ at the time of the December 2024 update"* | — | STMF: *"2000-2025 data is final"* |
| **DNK, FIN, ISL, NOR, SWE** | population registers (no census cycle → no post-censal revision) | ISL May-2026 revision touched 2022–24 population (non-binary redistribution, 0.04 %); ISL's Feb-2018 note is precedent that HMD does label an update *"provisional and will be revised once more detailed mortality data become available"* | STMF: FIN *"1990 to 2025 data are final"*; DNK *"2022 … 2026 are preliminary"*; ISL *"2020 to 2025 are preliminary"*; NOR *"2020-2026 data are preliminary"*; SWE *"2022 W52, 2023, 2024, 2025 and 2026 are preliminary"* — these refer to the weekly registration series; the annual HMD inputs come from the NSOs' final annual tabulations, so treat as a weak proxy |

Reading of the table: the 2024 **numerators** are final or near-final everywhere
except possibly TWN/CHL; the 2024 (and 2021–23) **denominators** are provisional in
every census-based population (USA, CHL, JPN, PRT, SVK, HRV, LTU, EST, LVA, LUX,
BEL, HKG, KOR, TWN) and effectively final in the register-based Nordics + CHE.

---

## C. Consequences for PREREGISTRATION.md

`PREREGISTRATION.md` is hash-stamped; none of this requires editing it. Record the
following as a dated **addendum** (new file or appended "Amendments" section with
its own hash), before any placebo or shift-regime real-data run.

**Placebo regime (A):**

1. **Keep all 11 populations.** Nothing found warrants EXCLUDE; BEL's exclusion is
   confirmed by HMD's own warning.
2. **Pre-register three placebo strata** and state which claims use which:
   neutral (CHE, DNK, FIN, ISL, NLD, NOR, SWE) = "pandemic-only break", the primary
   placebo stratum and the proper analogue to COVID; belligerent-total (FRATNP,
   GBRTENW, ITA); belligerent-civilian (GBR_SCO). Pooled results are reported, but
   the *Twin Crises* claim ("does 1914–22 miscalibration predict 2020–24?") should
   be made on the neutral stratum.
3. **Sensitivity S-P1: drop GBR_SCO** (civilian series, HMD-flagged male denominator).
4. **H4 (age profile) in the belligerent stratum:** report male ages 18–45 in 1914–20
   separately, labelled as HMD reconstructions (Vallin–Meslé; Jdanov 2005; Jdanov
   2008). Do not let those cells drive the head-line "coverage failure concentrates
   at age X" statement.
5. **Counts vs rates:** DNK (+5.5 % territory at 1921) and FRATNP (1914−/1914+,
   1920−/1920+) have exposure jumps inside the window. Every metric is on rates or
   on counts *given HMD exposure*, so no adjustment — but e₀/annuity evaluation must
   use the post-change exposure, and the runner must not interpolate exposures across
   those years.
6. **ITA training window:** HMD says ≤1905 "extra caution". The prereg admits
   1872–1913; either keep it and say so, or pre-specify 5-year age aggregation for
   ITA-specific figures. Decide now, in the addendum.
7. **Data appendix sentences to add:** (a) the MP Appendix A flu-triangle note; (b)
   NLD age-0 harmonisation at 1918; (c) FIN 1918 = civil war + flu, domestic deaths;
   (d) `<CC>note.pdf` files not consulted (login-walled).

**Shift regime (B):**

8. **Freeze and cite the vintage:** DOI `10.4054/HMD.Countries.20260615`, download
   date, and `data/MANIFEST.sha256` — already the prereg's vintage; add the DOI
   string to the addendum and to the paper's data statement.
9. **The pre-registered "drop 2024" sensitivity is necessary but mis-targeted as
   worded.** Keep it (S-S1: score h=1…4 only), and add **S-S2: drop USA and CHL 2024**
   (the two populations with explicit HMD 2024 caveats — USA's two-pass update and
   CHL's exposures awaiting the 2024 census). Optionally S-S3: register-based subset
   (CHE, DNK, FIN, ISL, NOR, SWE) vs census-based subset, since the revision risk is
   a denominator-vintage effect on 2021–24, not a 2024-only effect.
10. **Do not claim "2024 is final".** Correct wording: "2024 death counts are final at
    source for all populations except TWN and CHL (provisional per STMF metadata);
    2021–2024 exposures are post-censal and, per HMD Methods Protocol §5.2.3,
    provisional in census-based populations."
11. **Vintage-drift check at revision time:** if the paper is revised after a later
    HMD release, re-score the shift regime on the new vintage and report the delta
    as a robustness line rather than silently updating numbers; the DOI makes the
    two vintages citable side by side.

---

### Source URLs

- Country pages: `https://www.mortality.org/Country/Country?cntr=<CC>` for CHE, DNK,
  FIN, FRATNP, FRACNP, GBRTENW, GBRCENW, GBR_SCO, ISL, ITA, NLD, NOR, SWE
- Background PDFs: `https://www.mortality.org/File/GetDocument/hmd.v6/<CC>/Public/InputDB/<CC>com.pdf`
  (all 11 placebo + 14 remaining shift populations fetched 2026-08-26)
- Methods Protocol v6: `https://www.mortality.org/File/GetDocument/Public/Docs/MethodsProtocolV6.pdf`
- FAQ: `https://www.mortality.org/Project/FAQ` · What's New: `https://www.mortality.org/Data/WhatsNew`
- Citation Guidelines: `https://www.mortality.org/Research/CitationGuidelines`
- Zipped Data Files (DOI): `https://www.mortality.org/Data/ZippedDataFiles`
- STMF metadata: `https://www.mortality.org/File/GetDocument/Public/STMF/DOC/STMFmetadata.pdf`
- Login-walled, not consulted: `https://www.mortality.org/File/GetDocument/hmd.v6/<CC>/InputDB/<CC>note.pdf`
