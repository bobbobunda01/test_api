import json
import logging
import os
import pathlib
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from joblib import load
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed

from fichier_py.fonction import (
    prepare_input_features_enriched,
    predict_match_with_proba,
    log_prediction,
    get_valid_date,
    entree_utilisateur,
    get_last5_results_pattern,
    apply_unexpected_layer,
    resolve_fixture_id_local,
    explanation_from_pred_final,
    clean_extract_final_result,
)

app = Flask(__name__)
RACINE_PROJET = pathlib.Path(__file__).resolve().parents[1]
######------------------
######### SDDDE
# -------------------------------------------------------------------
# CONFIG GLOBALE
# -------------------------------------------------------------------

DEBUG_PREDICT = os.getenv("DEBUG_PREDICT", "0") == "1"
FAST_API_MODE = os.getenv("FAST_API_MODE", "0") == "1"

# Charge une seule fois
FIX_DATASET = pd.read_csv(RACINE_PROJET / "data" / "fix_id_dataset.csv")

# Cache mémoire des ressources par compétition
COMP_CACHE: Dict[int, Dict[str, Any]] = {}

app = Flask(__name__)
RACINE_PROJET = pathlib.Path(__file__).resolve().parents[1]

# -------------------------------------------------------------------
# CONFIG GLOBALE
# -------------------------------------------------------------------

DEBUG_PREDICT = os.getenv("DEBUG_PREDICT", "0") == "1"
FAST_API_MODE = os.getenv("FAST_API_MODE", "0") == "1"
#MatchInput="MatchInput"
# Charge une seule fois
FIX_DATASET = pd.read_csv(RACINE_PROJET / "data" / "fix_id_dataset.csv")

# Cache mémoire des ressources par compétition
COMP_CACHE: Dict[int, Dict[str, Any]] = {}

class MatchInput(BaseModel):
    HomeTeam: str
    AwayTeam: str
    comp: int
    odds_home: float
    odds_draw: float
    odds_away: float
    match_Date: str


class RequestBody(BaseModel):
    matches: List[MatchInput]


# -------------------------------------------------------------------
# TABLE DE CONFIGURATION DES COMPETITIONS
# -------------------------------------------------------------------

COMP_CONFIG = {
    39: {
        "key": "pl",
        "thread": 0.63,
        "current_df": "pl/saison_encours.csv",
        "season_prev": "pl/pl_24_25.csv",
        "past": {
            2022: "pl/premier_league_season_2022.csv",
            2023: "pl/premier_league_season_2023.csv",
            2024: "pl/premier_league_season_2024.csv",
            2025: "pl/premier_league_season_2025.csv",
        },
        "models": {
            "stage1": "pl/rf_pl_stage1.joblib",
            "stage2": "pl/rf_pl_stage2.joblib",
            "goals": "pl/xgboost_nbre_but_marque_pl.joblib",
        },
    },
    135: {
        "key": "sa1",
        "thread": 0.63,
        "current_df": "sa1/saison_encours.csv",
        "season_prev": "sa1/sa_24_25.csv",
        "past": {
            2022: "sa1/serie_a_season_2022.csv",
            2023: "sa1/serie_a_season_2023.csv",
            2024: "sa1/serie_a_season_2024.csv",
            2025: "sa1/serie_a_season_2025.csv",
        },
        "models": {
            "stage1": "sa1/rf_sa1_stage1.joblib",
            "stage2": "sa1/rf_sa1_stage2.joblib",
            "goals": "sa1/xgboost_nbre_but_marque_sa1.joblib",
        },
    },
    140: {
        "key": "lg1",
        "thread": 0.50,
        "current_df": "lg1/saison_encours.csv",
        "season_prev": "lg1/lg_24_25.csv",
        "past": {
            2022: "lg1/la_liga_season_2022.csv",
            2023: "lg1/la_liga_season_2023.csv",
            2024: "lg1/la_liga_season_2024.csv",
            2025: "lg1/la_liga_season_2025.csv",
        },
        "models": {
            "stage1": "lg1/lg_bl1_stage1.joblib",
            "stage2": "lg1/rf_bl1_stage2.joblib",
            "goals": "lg1/xgboost_nbre_but_marque_lg.joblib",
        },
    },
    78: {
        "key": "bl1",
        "thread": 0.65,
        "current_df": "bl1/saison_encours.csv",
        "season_prev": "bl1/bl_24_25.csv",
        "past": {
            2022: "bl1/bundesliga_season_2022.csv",
            2023: "bl1/bundesliga_season_2023.csv",
            2024: "bl1/bundesliga_season_2024.csv",
            2025: "bl1/bundesliga_season_2025.csv",
        },
        "models": {
            "stage1": "bl1/rf_bl1_stage1.joblib",
            "stage2": "bl1/rf_bl1_stage2.joblib",
            "goals": "bl1/xgboost_nbre_but_marque_bl1.joblib",
        },
    },
    61: {
        "key": "fl",
        "thread": 0.62,
        "current_df": "fl/saison_encours.csv",
        "season_prev": "fl/fl_24_25.csv",
        "past": {
            2022: "fl/ligue_1_season_2022.csv",
            2023: "fl/ligue_1_season_2023.csv",
            2024: "fl/ligue_1_season_2024.csv",
            2025: "fl/ligue_1_season_2025.csv",
        },
        "models": {
            "stage1": "fl/rf_stage1.joblib",
            "stage2": "fl/rf_stage2.joblib",
            "goals": "fl/xgboost_nbre_but_marque_fl.joblib",
        },
    },
    88: {
        "key": "N1",
        "thread": 0.63,
        "current_df": "N1/saison_encours.csv",
        "season_prev": "N1/N_24_25.csv",
        "past": {
            2022: "N1/neerdeland_season_2022.csv",
            2023: "N1/neerdeland_season_2023.csv",
            2024: "N1/neerdeland_season_2024.csv",
            2025: "N1/neerdeland_season_2025.csv",
        },
        "models": {
            "stage1": "N1/rf_N1_stage1.joblib",
            "stage2": "N1/rf_N1_stage2.joblib",
            "goals": "N1/rf_nbre_but_marque_autre.joblib",
        },
    },
    207: {
        "key": "sui",
        "thread": 0.60,
        "current_df": "sui/saison_encours.csv",
        "season_prev": "sui/suisse_2024_2025.csv",
        "past": {
            2022: "sui/suisse_season_2022.csv",
            2023: "sui/suisse_season_2023.csv",
            2024: "sui/suisse_season_2024.csv",
            2025: "sui/suisse_season_2025.csv",
        },
        "models": {
            "stage1": "sui/rf_stage1.joblib",
            "stage2": "sui/rf_stage2.joblib",
            "goals": "sui/xgboost_nbre_but_marque_sui.joblib",
        },
    },
    94: {
        "key": "port",
        "thread": 0.60,
        "current_df": "port/saison_encours.csv",
        "season_prev": "port/port_24_25.csv",
        "past": {
            2022: "port/portugais_season_2022.csv",
            2023: "port/portugais_season_2023.csv",
            2024: "port/portugais_season_2024.csv",
            2025: "port/portugais_season_2025.csv",
        },
        "models": {
            "stage1": "port/rf_stage1.joblib",
            "stage2": "port/rf_stage2.joblib",
            "goals": "port/xgboost_nbre_but_marque_port.joblib",
        },
    },
    203: {
        "key": "turk",
        "thread": 0.60,
        "current_df": "turk/saison_encours.csv",
        "season_prev": "turk/turk_24_25.csv",
        "past": {
            2022: "turk/turquie_season_2022.csv",
            2023: "turk/turquie_season_2023.csv",
            2024: "turk/turquie_season_2024.csv",
            2025: "turk/turquie_season_2025.csv",
        },
        "models": {
            "stage1": "turk/rf_stage1.joblib",
            "stage2": "turk/rf_stage2.joblib",
            "goals": "turk/xgboost_nbre_but_marque_turk.joblib",
        },
    },
    98: {
        "key": "japon",
        "thread": 0.60,
        "current_df": "japon/saison_encours.csv",
        "season_prev": "japon/japon_2024.csv",
        "past": {
            2022: "japon/japon_season_2022.csv",
            2023: "japon/japon_season_2023.csv",
            2024: "japon/japon_season_2024.csv",
            2025: "japon/japon_season_2025.csv",
        },
        "models": {
            "stage1": "japon/rf_stage1.joblib",
            "stage2": "japon/rf_stage2.joblib",
            "goals": "japon/xgboost_nbre_but_marque_japon.joblib",
        },
    },
    197: {
        "key": "grece",
        "thread": 0.63,
        "current_df": "grece/saison_encours.csv",
        "season_prev": "grece/grec_2024_25.csv",
        "past": {
            2022: "grece/grece_season_2022.csv",
            2023: "grece/grece_season_2023.csv",
            2024: "grece/grece_season_2024.csv",
            2025: "grece/grece_season_2025.csv",
        },
        "models": {
            "stage1": "grece/rf_stage1.joblib",
            "stage2": "grece/rf_stage2.joblib",
            "goals": "grece/xgboost_nbre_but_marque_grece.joblib",
        },
    },
    144: {
        "key": "belg",
        "thread": 0.63,
        "current_df": "belg/saison_encours.csv",
        "season_prev": "belg/belg_24_25.csv",
        "past": {
            2022: "belg/belgique_season_2022.csv",
            2023: "belg/belgique_season_2023.csv",
            2024: "belg/belgique_season_2024.csv",
            2025: "belg/belgique_season_2025.csv",
        },
        "models": {
            "stage1": "belg/rf_stage1.joblib",
            "stage2": "belg/rf_stage2.joblib",
            "goals": "belg/xgboost_nbre_but_marque_belg.joblib",
        },
    },
    71: {
        "key": "bresil",
        "thread": 0.62,
        "current_df": "bresil/saison_encours.csv",
        "season_prev": "bresil/bresil_2024.csv",
        "past": {
            2022: "bresil/bresil_season_2022.csv",
            2023: "bresil/bresil_season_2023.csv",
            2024: "bresil/bresil_season_2024.csv",
            2025: "bresil/bresil_season_2025.csv",
        },
        "models": {
            "stage1": "bresil/rf_stage1.joblib",
            "stage2": "bresil/rf_stage2.joblib",
            "goals": "bresil/xgboost_nbre_but_marque_bresil.joblib",
        },
    },
    179: {
        "key": "ecosse",
        "thread": 0.63,
        "current_df": "ecosse/saison_encours.csv",
        "season_prev": "ecosse/ecosse_2024_25.csv",
        "past": {
            2022: "ecosse/ecosse_season_2022.csv",
            2023: "ecosse/ecosse_season_2023.csv",
            2024: "ecosse/ecosse_season_2024.csv",
            2025: "ecosse/ecosse_season_2025.csv",
        },
        "models": {
            "stage1": "ecosse/rf_stage1_ecosse.joblib",
            "stage2": "ecosse/rf_stage2_ecosse.joblib",
            "goals": "ecosse/xgboost_nbre_but_marque_ecosse.joblib",
        },
    },
    119: {
        "key": "danemark",
        "thread": 0.63,
        "current_df": "danemark/saison_encours.csv",
        "season_prev": "danemark/Danemark_2024_25.csv",
        "past": {
            2022: "danemark/danemark_season_2022.csv",
            2023: "danemark/danemark_season_2023.csv",
            2024: "danemark/danemark_season_2024.csv",
            2025: "danemark/danemark_season_2025.csv",
        },
        "models": {
            "stage1": "danemark/rf_stage1.joblib",
            "stage2": "danemark/rf_stage2.joblib",
            "goals": "danemark/xgboost_nbre_but_marque_danemark.joblib",
        },
    },
    180: {
        "key": "ecosse_div_1",
        "thread": 0.64,
        "current_df": "ecosse_div_1/saison_encours.csv",
        "season_prev": "ecosse_div_1/ecosse_div_1_2024_2025.csv",
        "past": {
            2022: "ecosse_div_1/ecosse_div_1_season_2022.csv",
            2023: "ecosse_div_1/ecosse_div_1_season_2023.csv",
            2024: "ecosse_div_1/ecosse_div_1_season_2024.csv",
            2025: "ecosse_div_1/ecosse_div_1_season_2025.csv",
        },
        "models": {
            "stage1": "ecosse_div_1/rf_stage1_ecosse_div_1.joblib",
            "stage2": "ecosse_div_1/rf_stage2_ecosse_div_1.joblib",
            "goals": "ecosse_div_1/xgboost_nbre_but_marque_ecosse_div_1.joblib",
        },
    },
    235: {
        "key": "russie",
        "thread": 0.60,
        "current_df": "russie/saison_encours.csv",
        "season_prev": "russie/russie_2024_25.csv",
        "past": {
            2022: "russie/russie_season_2022.csv",
            2023: "russie/russie_season_2023.csv",
            2024: "russie/russie_season_2024.csv",
            2025: "russie/russie_season_2025.csv",
        },
        "models": {
            "stage1": "russie/rf_stage1_rus.joblib",
            "stage2": "russie/rf_stage2_rus.joblib",
            "goals": "russie/xgboost_nbre_but_marque_russie.joblib",
        },
    },
    292: {
        "key": "coree_sud",
        "thread": 0.60,
        "current_df": "coree_sud/saison_encours.csv",
        "season_prev": "coree_sud/df_coree_sud_2024.csv",
        "past": {
            2022: "coree_sud/df_coree_sud_2022.csv",
            2023: "coree_sud/df_coree_sud_2023.csv",
            2024: "coree_sud/df_coree_sud_2024.csv",
            2025: "coree_sud/df_coree_sud_2025.csv",
        },
        "models": {
            "stage1": "coree_Sud/coree_sud_stage1.joblib",
            "stage2": "coree_Sud/coree_sud_stage2.joblib",
            "goals": "coree_Sud/rf_nbre_but_marque_coree_sud.joblib",
        },
    },
    128: {
        "key": "argentine",
        "thread": 0.63,
        "current_df": "argentine/saison_encours.csv",
        "season_prev": "argentine/df_argentine_league1_2024.csv",
        "past": {
            2022: "argentine/df_argentine_league1_2022.csv",
            2023: "argentine/df_argentine_league1_2023.csv",
            2024: "argentine/df_argentine_league1_2024.csv",
            2025: "argentine/df_argentine_league1_2025.csv",
        },
        "models": {
            "stage1": "argentine/argentine_stage1.joblib",
            "stage2": "argentine/argentine_stage2.joblib",
            "goals": "argentine/rf_nbre_but_marque_argentine.joblib",
        },
    },
    3: {
        "key": "leagues_europa",
        "thread": 0.65,
        "current_df": "leagues_europa/saison_encours.csv",
        "season_prev": "leagues_europa/df_league_europa_2024.csv",
        "past": {
            2022: "leagues_europa/df_league_europa_2022.csv",
            2023: "leagues_europa/df_league_europa_2023.csv",
            2024: "leagues_europa/df_league_europa_2024.csv",
            2025: "leagues_europa/df_league_europa_2025.csv",
        },
        "models": {
            "stage1": "leagues_europa/league_europa_stage1.joblib",
            "stage2": "leagues_europa/league_europa_stage2.joblib",
            "goals": "leagues_europa/rf_nbre_but_marque_league_europa.joblib",
        },
    },
    2: {
        "key": "leagues_champions",
        "thread": 0.65,
        "current_df": "leagues_champions/saison_encours.csv",
        "season_prev": "leagues_champions/df_champions_2024.csv",
        "past": {
            2022: "leagues_champions/df_champions_2022.csv",
            2023: "leagues_champions/df_champions_2023.csv",
            2024: "leagues_champions/df_champions_2024.csv",
            2025: "leagues_champions/df_champions_2025.csv",
        },
        "models": {
            "stage1": "leagues_champions/league_champion_stage1.joblib",
            "stage2": "leagues_champions/league_champion_stage2.joblib",
            "goals": "leagues_champions/rf_nbre_but_marque_chamions_league.joblib",
        },
    },
    233: {
        "key": "egypte",
        "thread": 0.64,
        "current_df": "egypte/saison_encours.csv",
        "season_prev": "egypte/df_egypte_2024.csv",
        "past": {
            2022: "egypte/df_egypte_2022.csv",
            2023: "egypte/df_egypte_2023.csv",
            2024: "egypte/df_egypte_2024.csv",
            2025: "egypte/df_egypte_2025.csv",
        },
        "models": {
            "stage1": "egypte/egypte_stage1.joblib",
            "stage2": "egypte/egypte_stage2.joblib",
            "goals": "egypte/rf_nbre_but_marque_egypte.joblib",
        },
    },
    262: {
        "key": "mexique",
        "thread": 0.63,
        "current_df": "mexique/saison_encours.csv",
        "season_prev": "mexique/df_mexique_2024.csv",
        "past": {
            2022: "mexique/df_mexique_2022.csv",
            2023: "mexique/df_mexique_2023.csv",
            2024: "mexique/df_mexique_2024.csv",
            2025: "mexique/df_mexique_2025.csv",
        },
        "models": {
            "stage1": "mexique/mexique_stage1.joblib",
            "stage2": "mexique/mexique_stage2.joblib",
            "goals": "mexique/rf_nbre_but_marque_mexique.joblib",
        },
    },
    79: {
        "key": "bl2",
        "thread": 0.63,
        "current_df": "bl2/saison_encours.csv",
        "season_prev": "bl2/bl2_2024.csv",
        "past": {
            2022: "bl2/bl2_2022.csv",
            2023: "bl2/bl2_2023.csv",
            2024: "bl2/bl2_2024.csv",
            2025: "bl2/bl2_2025.csv",
        },
        "models": {
            "stage1": "bl2/bundesliga2_stage1.joblib",
            "stage2": "bl2/bundesliga2_stage2.joblib",
            "goals": "bl2/rf_nbre_but_marque_bundesliga2.joblib",
        },
    },
    136: {
        "key": "sa2",
        "thread": 0.67,
        "current_df": "sa2/saison_encours.csv",
        "season_prev": "sa2/serie_B_2024.csv",
        "past": {
            2022: "sa2/serie_B_2022.csv",
            2023: "sa2/serie_B_2023.csv",
            2024: "sa2/serie_B_2024.csv",
            2025: "sa2/serie_B_2025.csv",
        },
        "models": {
            "stage1": "sa2/serieB_stage1.joblib",
            "stage2": "sa2/serieB_stage1.joblib",
            "goals": "sa2/rf_nbre_but_marque_serieB.joblib",
        },
    },
    40: {
        "key": "pl2",
        "thread": 0.60,
        "current_df": "pl2/saison_encours.csv",
        "season_prev": "pl2/champioship_2024.csv",
        "past": {
            2022: "pl2/champioship_2022.csv",
            2023: "pl2/champioship_2023.csv",
            2024: "pl2/champioship_2024.csv",
            2025: "pl2/champioship_2025.csv",
        },
        "models": {
            "stage1": "pl2/pl2_stage1.joblib",
            "stage2": "pl2/pl2_stage2.joblib",
            "goals": "pl2/rf_nbre_but_marque_championship.joblib",
        },
    },
    62: {
        "key": "fl2",
        "thread": 0.63,
        "current_df": "fl2/saison_encours.csv",
        "season_prev": "fl2/fl2_2024.csv",
        "past": {
            2022: "fl2/fl2_2022.csv",
            2023: "fl2/fl2_2023.csv",
            2024: "fl2/fl2_2024.csv",
            2025: "fl2/fl2_2025.csv",
        },
        "models": {
            "stage1": "fl2/fl2_stage1.joblib",
            "stage2": "fl2/fl2_stage2.joblib",
            "goals": "fl2/rf_nbre_but_marque_fl2.joblib",
        },
    },
    141: {
        "key": "lg2",
        "thread": 0.63,
        "current_df": "lg2/saison_encours.csv",
        "season_prev": "lg2/lg2_2024.csv",
        "past": {
            2022: "lg2/lg2_2022.csv",
            2023: "lg2/lg2_2023.csv",
            2024: "lg2/lg2_2024.csv",
            2025: "lg2/lg2_2025.csv",
        },
        "models": {
            "stage1": "lg2/lg2_stage1.joblib",
            "stage2": "lg2/lg2_stage2.joblib",
            "goals": "lg2/rf_nbre_but_marque_lg2.joblib",
        },
    },
    6: {
        "key": "can",
        "thread": 0.65,
        "current_df": "can/saison_encours.csv",
        "season_prev": "can/df_can_2023.csv",
        "past": {
            2022: "can/df_can_2019.csv",
            2023: "can/df_can_2021.csv",
            2024: "can/df_can_2023.csv",
            2025: "can/df_can_2025.csv",
        },
        "models": {
            "stage1": "can/rf_stage1_can.joblib",
            "stage2": "can/rf_stage2_can.joblib",
            "goals": "can/xgboost_nbre_but_marque_can.joblib",
        },
    },
}

# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _read_csv_with_date(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def _load_comp_resources(comp: int) -> Dict[str, Any]:
    """
    Charge datasets + modèles une seule fois par compétition,
    puis les garde en mémoire.
    """
    if comp in COMP_CACHE:
        return COMP_CACHE[comp]

    cfg = COMP_CONFIG.get(comp)
    if cfg is None:
        raise ValueError(f"Compétition non supportée: {comp}")

    current_df = _read_csv_with_date(RACINE_PROJET / "data" / cfg["current_df"])
    season_prev = _read_csv_with_date(RACINE_PROJET / "data" / cfg["season_prev"])

    past_dfs = {}
    for year, rel_path in cfg["past"].items():
        past_dfs[year] = _read_csv_with_date(RACINE_PROJET / "data" / rel_path)

    model_stage1 = load(RACINE_PROJET / "modele" / cfg["models"]["stage1"])
    model_stage2 = load(RACINE_PROJET / "modele" / cfg["models"]["stage2"])
    model_goals = load(RACINE_PROJET / "modele" / cfg["models"]["goals"])

    resources = {
        "thread": cfg["thread"],
        "df": current_df,
        "season_preced": season_prev,
        "df_2022": past_dfs.get(2022),
        "df_2023": past_dfs.get(2023),
        "df_2024": past_dfs.get(2024),
        "df_2025": past_dfs.get(2025),
        "modele1": model_stage1,
        "modele2": model_stage2,
        "model_but": model_goals,
    }

    COMP_CACHE[comp] = resources
    return resources


def _build_match_result(match: MatchInput) -> Dict[str, Any]:
    """
    Traitement d'un match.
    Cette fonction ne recharge rien de lourd.
    """
    home = match.HomeTeam
    away = match.AwayTeam
    comp = int(match.comp)
    odds_h = float(match.odds_home)
    odds_d = float(match.odds_draw)
    odds_a = float(match.odds_away)
    match_date = match.match_Date

    res = _load_comp_resources(comp)

    df = res["df"]
    season_preced = res["season_preced"]
    df_2022 = res["df_2022"]
    df_2023 = res["df_2023"]
    df_2024 = res["df_2024"]
    df_2025 = res["df_2025"]
    modele1 = res["modele1"]
    modele2 = res["modele2"]
    model_but = res["model_but"]
    thread = res["thread"]

    date_match = get_valid_date(match_date)

    features_input = prepare_input_features_enriched(
        home, away, date_match, odds_h, odds_a, odds_d, df, league_code=comp
    )
    features_input = features_input.copy()
    features_input["home"] = str(home)
    features_input["away"] = str(away)
    features_input["match_date"] = str(date_match)

    if DEBUG_PREDICT:
        log_prediction(features_input.to_json())

    X_inputs = entree_utilisateur(home, away, odds_h, odds_d, odds_a, df, season_preced)

    perf_home = get_last5_results_pattern(df, home, date_match)
    perf_away = get_last5_results_pattern(df, away, date_match)

    fid = resolve_fixture_id_local(FIX_DATASET, home, away, match_date, league_code=comp)
    if fid is not None:
        features_input["fixture_id"] = int(fid)

    features_input["_use_realtime"] = True

    pred = predict_match_with_proba(
        features_input,
        model_stage1=modele1,
        model_stage2=modele2,
        threshold_draw=thread,
        league_code=comp,
    )

    pred_final = apply_unexpected_layer(
        base_pred=pred,
        season_current_df=df,
        season_past_list=[df_2022, df_2023, df_2024, df_2025],
        home=home,
        away=away,
        match_date=date_match,
        feats_df=features_input,
        league_code=comp,
        X_ref_features=None,
    )

    if not FAST_API_MODE:
        pred_final = explanation_from_pred_final(pred_final, user_profile="standard")

    pred_but = model_but.predict(X_inputs)[0]
    mess_but = (
        "✅ Prédiction :",
        "Plus de buts en 2ᵉ mi-temps" if pred_but == 1 else "Plus de buts en 1ʳᵉ mi-temps",
    )

    pred_final["home"] = home
    pred_final["away"] = away
    pred_final["5_dern_perf_home"] = np.array(perf_home).item()
    pred_final["5_dern_perf_away"] = np.array(perf_away).item()
    pred_final["plus_but"] = int(pred_but)
    pred_final["mess_but"] = str(mess_but)
    final_json = clean_extract_final_result(pred_final)

    return final_json

# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------

@app.route("/", methods=["GET"])
def accueil():
    return jsonify({"Message": "Bienvenue sur l'API de prédiction de matchs"})


@app.route("/predire/pl", methods=["POST"])

def prediction():
    if not request.json:
        return jsonify({"Erreur": "Aucun fichier JSON fourni"}), 400

    try:
        body = RequestBody(**request.json)

        matches = body.matches or []
        if not matches:
            return jsonify({"Erreur": "La liste des matchs est vide"}), 400

        # Précharge une seule fois les ressources par compétition
        needed_comps = sorted({int(m.comp) for m in matches})
        for comp in needed_comps:
            _load_comp_resources(comp)

        all_results = [None] * len(matches)
        errors = []

        max_workers = min(4, len(matches))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_build_match_result, match): idx
                for idx, match in enumerate(matches)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                match = matches[idx]

                try:
                    all_results[idx] = future.result()
                except Exception as e:
                    errors.append({
                        "index": idx,
                        "home": getattr(match, "HomeTeam", None),
                        "away": getattr(match, "AwayTeam", None),
                        "comp": getattr(match, "comp", None),
                        "error": f"{type(e).__name__}: {str(e)[:300]}"
                    })

                    all_results[idx] = {
                        "home": getattr(match, "HomeTeam", None),
                        "away": getattr(match, "AwayTeam", None),
                        "prediction": None,
                        "prediction_model": None,
                        "proba_0": None,
                        "proba_1": None,
                        "proba_2": None,
                        "double_chance": None,
                        "bias_detected": None,
                        "low_confidence": None,
                        "realtime_risk": {
                            "available": False,
                            "fixture_id": getattr(match, "fixture_id", None),
                            "missing": [f"parallel_match_error:{type(e).__name__}"],
                            "reasons": [f"parallel_match_error:{type(e).__name__}"],
                            "risk_level": "UNKNOWN",
                            "risk_score": 0.0,
                            "summary": {}
                        },
                        "explanation": "Analyse indisponible pour ce match.",
                        "notes": [f"parallel_match_error:{type(e).__name__}"]
                    }

        response = {"Resultats": all_results}
        if errors:
            response["warnings"] = errors

        return jsonify(response)

    except Exception as e:
        return jsonify({"Erreur": f"{type(e).__name__}: {str(e)}"}), 400
    
if __name__ == '__main__':
    app.run(debug=True)