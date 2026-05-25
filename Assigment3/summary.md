## 1. Initial Data Analysis
The first step is to look at the data from the four files to understand fields and encoding differences.

| Field | Dayzer | Pano | Observations |
| :--- | :--- | :--- | :--- |
| **Bus Name** | KEITHSW | LVOAK_138KV_1, ADK_345KV_2 | Pano encodes voltage + section in the name; Dayzer stores voltage separately |
| **Voltage** | `BASKV` column | `BASKV` column **AND** in name | Dual encoding in Pano — easy cross-check |
| **Coordinates** | LAT/LON | LAT/LON | Same projection (WGS84-looking decimal degrees) |
| **Bus Type** | Absent | LOAD/GEN/ISOLATED | Pano-only signal |
| **Station** | Absent | STATION (e.g. ADK) | Pano groups multiple bus sections under one parent substation |
| **Area** | BEPC_TSP | ERC | Different vocabularies for the same region |
| **Graph Edge** | FROM_BUS / TO_BUS (names) | FROM_BUS_ID / TO_BUS_ID (numeric) + names | **Topology is the killer signal** — see below |

### The 1:N Mapping Problem
The single most important observation up front: Pano splits a station into multiple bus sections (`ADK_138KV_1`, `ADK_138KV_2`, `ADK_138KV_3`) while Dayzer typically does not. 

I realized that this isn't just a fuzzy-string problem because we are given coordinates and graph. So this is not a 1:1 problem — it's **1:N**, and any pipeline that assumes 1:1 will silently corrupt half the matches. The deliverable schema needs to express that, e.g. one row per (Dayzer bus, Pano bus) candidate pair with a confidence.

---

## 2. Proposed Matching Pipeline
I realized that this isn't just a fuzzy-string problem because we are given coordinates and graph data.

### Step 1: Spatial Bucketing
Bucket buses by (`voltage_rounded`, `geo_cell` ~5km). A 5 km geohash cell at the same voltage typically holds <10 candidates. This cuts the comparison space from $N^2$ to roughly $N \log N$.

### Step 2: Geo-Exact Anchoring
Inside each block, pair buses where **Haversine distance < 200 m** AND **voltage matches within 1 kV**. This anchors the unambiguous core — probably 70–85% of buses. 
* **Tag method:** `geo_exact`
* **Confidence:** 1.0

### Step 3: Name Cleaning for 1:N Matches
The hard case: a Pano station with three sections (`ADK_138KV_1/2/3`) all at the same coords ±50 m. 
* Strip the voltage/section tokens out of the Pano name (`ADK_138KV_1` → `ADK`) and match against Dayzer's name. 
* **Tag method:** `geo+name`
* **Output:** ALL three as candidate 1:N matches if Dayzer collapsed them.

### Step 4: Graph Building
Build graphs from the branch tables: 
* **Nodes:** buses
* **Edges:** branches at matching voltage. 

### Step 5: Topological Validation
For each unmatched Dayzer bus $D$:
* Compute: of every Dayzer branch $D_1—D_2$, how often does the mapped pair $M(D_1)—M(D_2)$ exist as an edge in Pano, with similar $R/X/B$?

### Step 6: Iteration
Make sure the output is what you expected; if not, go back and redo some steps.
