-- --- STEP 1: Data Unification ---
-- Combines data from all five sheets into a single master dataset.
WITH CombinedData AS (
    SELECT AT, V, AP, RH, PE FROM Sheet1
    UNION ALL
    SELECT AT, V, AP, RH, PE FROM Sheet2
    UNION ALL
    SELECT AT, V, AP, RH, PE FROM Sheet3
    UNION ALL
    SELECT AT, V, AP, RH, PE FROM Sheet4
    UNION ALL
    SELECT AT, V, AP, RH, PE FROM Sheet5
),

-- --- STEP 2: Comprehensive Summary Statistics ---
-- Calculates global measures (mean, min, max, std dev) for all features.
SummaryStats AS (
    SELECT
        COUNT(*) AS total_rows,
        AVG(AT) AS avg_at, STDDEV(AT) AS stddev_at, MIN(AT) AS min_at, MAX(AT) AS max_at,
        AVG(V) AS avg_v, STDDEV(V) AS stddev_v, MIN(V) AS min_v, MAX(V) AS max_v,
        AVG(AP) AS avg_ap, STDDEV(AP) AS stddev_ap, MIN(AP) AS min_ap, MAX(AP) AS max_ap,
        AVG(RH) AS avg_rh, STDDEV(RH) AS stddev_rh, MIN(RH) AS min_rh, MAX(RH) AS max_rh,
        AVG(PE) AS avg_pe, STDDEV(PE) AS stddev_pe, MIN(PE) AS min_pe, MAX(PE) AS max_pe
    FROM CombinedData
),

-- --- STEP 3: Feature Engineering - Categorize Ambient Temperature (AT) ---
-- Creates discrete bins for AT to analyze non-linear effects, a crucial step
-- for exploratory analysis before non-linear models like XGBoost.
BinnedAT AS (
    SELECT
        AT, V, AP, RH, PE,
        CASE
            WHEN AT < 10.0 THEN '1_Cold (< 10°C)'
            WHEN AT >= 10.0 AND AT < 20.0 THEN '2_Mild (10°C - 20°C)'
            WHEN AT >= 20.0 AND AT < 30.0 THEN '3_Warm (20°C - 30°C)'
            ELSE '4_Hot (>= 30°C)'
        END AS AT_Category
    FROM CombinedData
),

-- --- STEP 4: Analysis by Temperature Category ---
-- Groups data by the new category and calculates mean PE/V/RH for each bin.
CategoryAnalysis AS (
    SELECT
        AT_Category,
        COUNT(*) AS Category_Count,
        AVG(AT) AS Avg_AT_in_Category,
        AVG(PE) AS Avg_PE_in_Category,
        AVG(V) AS Avg_V_in_Category,
        AVG(RH) AS Avg_RH_in_Category
    FROM BinnedAT
    GROUP BY AT_Category
    ORDER BY AT_Category
),

-- --- STEP 5: Outlier & Efficiency Ranking (Z-Scores and Deciles) ---
-- Uses window functions to calculate Z-Score for PE (for outlier detection)
-- and NTILE(10) to rank observations into deciles of efficiency.
ZScoresAndDeciles AS (
    SELECT
        AT, V, AP, RH, PE,
        -- Calculate PE Z-Score: (Value - Mean) / Standard Deviation
        (PE - (SELECT avg_pe FROM SummaryStats)) / (SELECT stddev_pe FROM SummaryStats) AS PE_ZScore,
        -- Rank PE into 10 deciles (1 is the highest efficiency/PE)
        NTILE(10) OVER (ORDER BY PE DESC) AS PE_Decile_Rank
    FROM CombinedData
),

-- --- STEP 6: High Efficiency (Top Decile) Analysis ---
-- Isolates the top 10% most efficient operating points (PE_Decile_Rank = 1)
-- to identify the optimal environmental conditions.
HighEfficiencyAnalysis AS (
    SELECT
        'Decile 1 (Highest PE)' AS Analysis_Group,
        AVG(AT) AS Avg_AT_High_Eff,
        AVG(V) AS Avg_V_High_Eff,
        AVG(AP) AS Avg_AP_High_Eff,
        AVG(RH) AS Avg_RH_High_Eff,
        AVG(PE) AS Avg_PE_High_Eff,
        COUNT(*) AS Count_High_Eff
    FROM ZScoresAndDeciles
    WHERE PE_Decile_Rank = 1
)

-- --- FINAL OUTPUT: Combining Results for Presentation ---
-- A complex UNION ALL structure is used to combine all analytical results
-- into a single output table, ensuring column counts are consistent with NULL padding.

SELECT '1. OVERALL SUMMARY STATISTICS' AS Analysis_Area,
       'Total Rows' AS Metric_A, CAST(total_rows AS VARCHAR) AS Value_A,
       'Avg PE' AS Metric_B, CAST(avg_pe AS VARCHAR) AS Value_B,
       'StdDev PE' AS Metric_C, CAST(stddev_pe AS VARCHAR) AS Value_C,
       'Min PE' AS Metric_D, CAST(min_pe AS VARCHAR) AS Value_D,
       'Max PE' AS Metric_E, CAST(max_pe AS VARCHAR) AS Value_E,
       'Avg AT' AS Metric_F, CAST(avg_at AS VARCHAR) AS Value_F,
       'Avg RH' AS Metric_G, CAST(avg_rh AS VARCHAR) AS Value_G
FROM SummaryStats

UNION ALL

SELECT '2. HIGH EFFICIENCY (TOP 10% PE) VS OVERALL AVERAGE',
       'Avg PE (High Eff)' AS Metric_A, CAST(Avg_PE_High_Eff AS VARCHAR) AS Value_A,
       'Overall Avg PE' AS Metric_B, CAST((SELECT avg_pe FROM SummaryStats) AS VARCHAR) AS Value_B,
       'PE Improvement (MW)' AS Metric_C, CAST((Avg_PE_High_Eff - (SELECT avg_pe FROM SummaryStats)) AS VARCHAR) AS Value_C,
       'Avg AT (High Eff)' AS Metric_D, CAST(Avg_AT_High_Eff AS VARCHAR) AS Value_D,
       'Avg V (High Eff)' AS Metric_E, CAST(Avg_V_High_Eff AS VARCHAR) AS Value_E,
       'Avg RH (High Eff)' AS Metric_F, CAST(Avg_RH_High_Eff AS VARCHAR) AS Value_F,
       'Avg AP (High Eff)' AS Metric_G, CAST(Avg_AP_High_Eff AS VARCHAR) AS Value_G
FROM HighEfficiencyAnalysis

UNION ALL

SELECT '3. ANALYSIS BY AMBIENT TEMPERATURE CATEGORY',
       AT_Category AS Metric_A, NULL AS Value_A,
       'Category Count' AS Metric_B, CAST(Category_Count AS VARCHAR) AS Value_B,
       'Avg PE' AS Metric_C, CAST(Avg_PE_in_Category AS VARCHAR) AS Value_C,
       'Avg AT' AS Metric_D, CAST(Avg_AT_in_Category AS VARCHAR) AS Value_D,
       'Avg V' AS Metric_E, CAST(Avg_V_in_Category AS VARCHAR) AS Value_E,
       'Avg RH' AS Metric_F, CAST(Avg_RH_in_Category AS VARCHAR) AS Value_F,
       'Avg AP' AS Metric_G, NULL AS Value_G
FROM CategoryAnalysis

UNION ALL

SELECT '4. TOP 5 POTENTIAL OUTLIERS (Highest PE/Z-Score)',
       'PE' AS Metric_A, CAST(PE AS VARCHAR) AS Value_A,
       'PE Z-Score' AS Metric_B, CAST(PE_ZScore AS VARCHAR) AS Value_B,
       'AT' AS Metric_C, CAST(AT AS VARCHAR) AS Value_C,
       'V' AS Metric_D, CAST(V AS VARCHAR) AS Value_D,
       'AP' AS Metric_E, CAST(AP AS VARCHAR) AS Value_E,
       'RH' AS Metric_F, CAST(RH AS VARCHAR) AS Value_F,
       'Decile' AS Metric_G, CAST(PE_Decile_Rank AS VARCHAR) AS Value_G
FROM ZScoresAndDeciles
ORDER BY PE_ZScore DESC
LIMIT 5;