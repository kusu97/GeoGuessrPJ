'''
This file is adapted from:
https://github.com/ccmdi/geobench/blob/main/geo2p/canon.py
Original author: ccmdi
License: MIT

A script for checking if two country names refer to the same canonical country.

Note: To accommodate the two-letter country codes (ISO 3166-1 alpha-2) used in the OSV-5M dataset, 
appropriate additions were made to COUNTRY_GROUPS. (see country_codes_in_osv5m.txt)
'''

# Map all variations to their canonical form in the dataset
COUNTRY_GROUPS = [
    # A
    ["Afghanistan", "islamic republic of afghanistan", "afg", "AF"],
    ["Albania", "republic of albania", "shqipëria", "shqiperia", "AL"],
    ["Algeria", "people's democratic republic of algeria", "al-jazā'ir", "al-jazair", "DZ"],
    ["Andorra", "principality of andorra"],
    ["Angola", "republic of angola", "AO"],
    ["Antigua and Barbuda", "antigua", "barbuda", "AG"],
    ["Argentina", "argentine republic", "república argentina", "republica argentina", "AR"],
    ["Armenia", "republic of armenia", "hayastan", "AM"],
    ["Australia", "commonwealth of australia", "aus", "straya", "aussie", "AU", "CX"],
    ["Austria", "republic of austria", "österreich", "osterreich", "AT"],
    ["Azerbaijan", "republic of azerbaijan", "azərbaycan", "azerbaycan", "AZ"],

    # B
    ["Bahamas (the)", "the bahamas", "bahamas", "commonwealth of the bahamas", "BS"],
    ["Bahrain", "kingdom of bahrain", "al-baḥrayn", "al-bahrayn", "BH"],
    ["Bangladesh", "people's republic of bangladesh", "BD"],
    ["Barbados", "bajan", "BB"],
    ["Belarus", "republic of belarus", "byelorussia", "belorussia", "BY"],
    ["Belgium", "kingdom of belgium", "belgië", "belgique", "BE"],
    ["Belize", "british honduras", "BZ"],
    ["Benin", "republic of benin", "dahomey", "BJ"],
    ["Bhutan", "kingdom of bhutan", "druk yul", "BT"],
    ["Bolivia (Plurinational State of)", "bolivia", "plurinational state of bolivia", "estado plurinacional de bolivia", "BO"],
    ["Bosnia and Herzegovina", "bosnia", "herzegovina", "bih", "bosna i hercegovina", "BA"],
    ["Botswana", "republic of botswana", "BW"],
    ["Brazil", "federative republic of brazil", "brasil", "república federativa do brasil", "BR"],
    ["Brunei Darussalam", "brunei", "nation of brunei", "brunei darussalam", "BN"],
    ["Bulgaria", "republic of bulgaria", "BG"],
    ["Burkina Faso", "burkina", "upper volta", "BF"],
    ["Burundi", "republic of burundi", "BI"],

    # C
    ["Cabo Verde", "cape verde", "republic of cabo verde", "CV"],
    ["Cambodia", "kingdom of cambodia", "kampuchea", "KH"],
    ["Cameroon", "republic of cameroon", "république du cameroun", "CM"],
    ["Canada", "dominion of canada", "can", "canuck", "CA"],
    ["Central African Republic (the)", "central african republic", "car", "the central african republic", "CF"],
    ["Chad", "republic of chad", "tchad", "TD"],
    ["Chile", "republic of chile", "república de chile", "CL"],
    ["China", "people's republic of china", "prc", "mainland china", "zhongguo", "zhōngguó",
     "Hong Kong", "hong kong sar", "hong kong special administrative region", "hksar", "hk", "HK",
     "Macao", "macau", "macao sar", "macau special administrative region",
     "Taiwan (Province of China)", "taiwan", "republic of china", "chinese taipei", "formosa", "TW", "CN"],
    ["Colombia", "republic of colombia", "república de colombia", "CO"],
    ["Comoros (the)", "comoros", "union of the comoros", "the comoros", "KM"],
    ["Congo (the)", "congo", "republic of the congo", "congo-brazzaville", "the congo", "CG"],
    ["Congo (the Democratic Republic of the)", "democratic republic of the congo", "drc", "dr congo", 
     "congo-kinshasa", "zaire", "the democratic republic of the congo", "CD"],
    ["Costa Rica", "republic of costa rica", "república de costa rica", "CR"],
    ["Côte d'Ivoire", "cote d'ivoire", "ivory coast", "republic of côte d'ivoire", "CI"],
    ["Croatia", "republic of croatia", "hrvatska", "HR"],
    ["Cuba", "republic of cuba", "república de cuba", "CU"],
    ["Cyprus", "republic of cyprus", "kypros", "kibris", "CY"],
    ["Czechia", "czech republic", "česko", "česká republika", "cesko", "CZ"],

    # D
    ["Denmark", "kingdom of denmark", "danmark",
     "Faroe Islands (the)", "faroe islands", "the faroe islands", "føroyar", "foroyar", "FO",
     "Greenland", "kalaallit nunaat", "GL", "DK"],
    ["Djibouti", "republic of djibouti", "DJ"],
    ["Dominica", "commonwealth of dominica", "DM"],
    ["Dominican Republic (the)", "dominican republic", "the dominican republic", "república dominicana", "DO"],

    # E
    ["Ecuador", "republic of ecuador", "república del ecuador", "EC"],
    ["Egypt", "arab republic of egypt", "misr", "مصر", "EG"],
    ["El Salvador", "republic of el salvador", "república de el salvador", "SV"],
    ["Equatorial Guinea", "republic of equatorial guinea", "GQ"],
    ["Eritrea", "state of eritrea", "ER"],
    ["Estonia", "republic of estonia", "eesti", "EE"],
    ["Eswatini", "kingdom of eswatini", "swaziland", "SZ"],
    ["Ethiopia", "federal democratic republic of ethiopia", "ET"],

    # F
    ["Fiji", "republic of fiji", "FJ"],
    ["Finland", "republic of finland", "suomi",
     "Åland Islands", "aland islands", "åland", "aland", "AX", "FI"],
    ["France", "french republic", "république française", "republique francaise", "FR",
     "French Polynesia", "polynésie française", "polynesie francaise", "PF",
     "New Caledonia", "nouvelle-calédonie", "nouvelle-caledonie", "NC",
     "French Guiana", "guyane", "guyane française", "guyane francaise", "GF",
     "Réunion", "reunion", "Réunion (France)", "île de la réunion", "ile de la reunion", "RE",
     "Martinique", "martinica", "MQ",
     "Guadeloupe", "guadalupe", "GP",
     "Saint Martin (French part)", "st martin", "saint-martin", "MF",
     "Saint Barthélemy", "st barthelemy", "saint-barthélemy", "saint-barthelemy",
     "Saint Pierre and Miquelon", "st pierre and miquelon", "saint-pierre et miquelon", 
     "Wallis and Futuna", "wallis-et-futuna", "wallis et futuna", "territory of the wallis and futuna islands", 
     "French Southern and Antarctic Lands", "french southern territories", 
     "terres australes et antarctiques françaises", "YT"],

    # G
    ["Gabon", "gabonese republic", "république gabonaise", "GA"],
    ["Gambia (the)", "gambia", "the gambia", "republic of the gambia", "GM"],
    ["Georgia", "საქართველო", "sakartvelo", "GE"],
    ["Germany", "federal republic of germany", "deutschland", "bundesrepublik deutschland", "DE"],
    ["Ghana", "republic of ghana", "GH"],
    ["Greece", "hellenic republic", "elláda", "ellada", "hellas", "GR"],
    ["Grenada", "spice isle", "GD"],
    ["Guatemala", "republic of guatemala", "república de guatemala", "GT"],
    ["Guinea", "republic of guinea", "guinée", "GN"],
    ["Guinea-Bissau", "republic of guinea-bissau", "GW"],
    ["Guyana", "co-operative republic of guyana", "GY"],

    # H
    ["Haiti", "republic of haiti", "république d'haïti", "république d'haiti", "HT"],
    ["Holy See (the)", "the holy see", "holy see", "vatican", "vatican city", "vatican city state", "VA"],
    ["Honduras", "republic of honduras", "república de honduras", "HN"],
    ["Hungary", "magyarország", "magyarorszag", "HU"],

    # I
    ["Iceland", "republic of iceland", "ísland", "island", "IS"],
    ["India", "republic of india", "bharat", "hindustan", "IN"],
    ["Indonesia", "republic of indonesia", "ID"],
    ["Iran (Islamic Republic of)", "iran", "islamic republic of iran", "persia", "IR"],
    ["Iraq", "republic of iraq", "IQ"],
    ["Ireland", "republic of ireland", "éire", "eire", "IE"],
    ["Israel", "state of israel", "yisra'el", "yisrael", "IL"],
    ["Italy", "italian republic", "italia", "repubblica italiana", "IT"],

    # J
    ["Jamaica", "jam", "JM"],
    ["Japan", "nippon", "nihon", "日本", "JP"],
    ["Jordan", "hashemite kingdom of jordan", "al-urdun", "JO"],

    # K
    ["Kazakhstan", "republic of kazakhstan", "qazaqstan", "KZ"],
    ["Kenya", "republic of kenya", "KE"],
    ["Kiribati", "republic of kiribati", "KI"],
    ["Korea (the Democratic People's Republic of)", "north korea", "democratic people's republic of korea", "dprk", "KP"],
    ["Korea (the Republic of)", "south korea", "republic of korea", "korea", "rok", "hanguk", "KR"],
    ["Kuwait", "state of kuwait", "KW"],
    ["Kyrgyzstan", "kyrgyz republic", "kirghizia", "KG"],
    ["Kosovo", "republic of kosovo", "XK"],

    # L
    ["Lao People's Democratic Republic (the)", "laos", "lao", "the lao people's democratic republic", "LA"],
    ["Latvia", "republic of latvia", "latvija", "LV"],
    ["Lebanon", "lebanese republic", "lubnan", "LB"],
    ["Lesotho", "kingdom of lesotho", "LS"],
    ["Liberia", "republic of liberia", "LR"],
    ["Libya", "state of libya", "libyan arab jamahiriya", "LY"],
    ["Liechtenstein", "principality of liechtenstein"],
    ["Lithuania", "republic of lithuania", "lietuva", "LT"],
    ["Luxembourg", "grand duchy of luxembourg", "letzebuerg", "LU"],

    # M
    ["Madagascar", "republic of madagascar", "malagasy republic", "MG"],
    ["Malawi", "republic of malawi", "nyasaland", "MW"],
    ["Malaysia", "mys", "MY"],
    ["Maldives", "republic of maldives", "dhivehi raajje", "MV"],
    ["Mali", "republic of mali", "ML"],
    ["Malta", "republic of malta", "MT"],
    ["Marshall Islands (the)", "marshall islands", "republic of the marshall islands", "the marshall islands"],
    ["Mauritania", "islamic republic of mauritania", "MR"],
    ["Mauritius", "republic of mauritius", "MU"],
    ["Mexico", "united mexican states", "méxico", "mexico", "estados unidos mexicanos", "MX"],
    ["Micronesia (Federated States of)", "micronesia", "federated states of micronesia", "fsm", "FM"],
    ["Moldova (the Republic of)", "moldova", "republic of moldova", "the republic of moldova", "MD"],
    ["Monaco", "principality of monaco"], 
    ["Mongolia", "mongol uls", "MN"],
    ["Montenegro", "crna gora", "ME"],
    ["Morocco", "kingdom of morocco", "al-maghrib", "MA"],
    ["Mozambique", "republic of mozambique", "moçambique", "MZ"],
    ["Myanmar", "republic of the union of myanmar", "burma", "MM"],

    # N
    ["Namibia", "republic of namibia", "southwest africa"],
    ["Nauru", "republic of nauru"], 
    ["Nepal", "federal democratic republic of nepal", "NP"],
    ["Netherlands (the)", "netherlands", "the netherlands", "holland", "nederland",
     "kingdom of the netherlands", "NL",
     "Aruba", "aw", "AW",
     "Curaçao", "curacao", "CW",
     "Sint Maarten (Dutch part)", "sint maarten", "saint martin (dutch part)", "SX",
     "Bonaire, Sint Eustatius and Saba", "bes islands", "BQ", 
     "caribbean netherlands"],
    ["New Zealand", "nz", "aotearoa", "Cook Islands", "kuki airani", "Niue", "Tokelau", "NZ"],
    ["Nicaragua", "republic of nicaragua", "república de nicaragua", "NI"],
    ["Niger (the)", "niger", "republic of the niger", "the niger", "NE"],
    ["Nigeria", "federal republic of nigeria", "NG"],
    ["North Macedonia", "republic of north macedonia", "macedonia", "fyrom", "former yugoslav republic of macedonia", "MK"],
    ["Norway", "kingdom of norway", "norge", "noreg", "NO", "SJ"],

    # O
    ["Oman", "sultanate of oman", "OM"],

    # P
    ["Pakistan", "islamic republic of pakistan", "PK"],
    ["Palau", "republic of palau", "belau", "PW"],
    ["Palestine, State of", "palestine", "state of palestine", "west bank and gaza", "palestinian territories", "PS"],
    ["Panama", "republic of panama", "república de panamá", "PA"],
    ["Papua New Guinea", "png", "papua", "independent state of papua new guinea", "PG"],
    ["Paraguay", "republic of paraguay", "república del paraguay", "PY"],
    ["Peru", "republic of peru", "república del perú", "republica del peru", "PE"],
    ["Philippines (the)", "philippines", "the philippines", "republic of the philippines", "pilipinas", "PH"],
    ["Poland", "republic of poland", "polska", "rzeczpospolita polska", "PL"],
    ["Portugal", "portuguese republic", "república portuguesa", "republica portuguesa", "PT"],

    # Q
    ["Qatar", "state of qatar", "QA"],

    # R
    ["Romania", "românia", "romania", "RO"],
    ["Russian Federation (the)", "russia", "russian federation", "the russian federation", "rossiya", "rossiyskaya federatsiya", "RU"],
    ["Rwanda", "republic of rwanda", "RW"],

    # S
    ["Saint Kitts and Nevis", "st. kitts and nevis", "st kitts and nevis", "KN"],
    ["Saint Lucia", "st. lucia", "st lucia", "LC"],
    ["Saint Vincent and the Grenadines", "st. vincent and the grenadines", "st vincent and the grenadines", "svg", "VC"],
    ["Samoa", "independent state of samoa", "western samoa", "WS"],
    ["San Marino", "republic of san marino", "serenissima repubblica di san marino"],
    ["Sao Tome and Principe", "são tomé and príncipe", "democratic republic of são tomé and príncipe", "ST"],
    ["Saudi Arabia", "kingdom of saudi arabia", "ksa", "saudi", "SA"],
    ["Senegal", "republic of senegal", "république du sénégal", "republique du senegal", "SN"],
    ["Serbia", "republic of serbia", "republika srbija", "RS"],
    ["Seychelles", "republic of seychelles"],
    ["Sierra Leone", "republic of sierra leone", "SL"],
    ["Singapore", "republic of singapore", "sing", "sg", "lion city", "SG"],
    ["Slovakia", "slovak republic", "slovensko", "SK"],
    ["Slovenia", "republic of slovenia", "slovenija", "SI"],
    ["Solomon Islands", "sol", "SB"],
    ["Somalia", "federal republic of somalia", "SO"],
    ["South Africa", "republic of south africa", "rsa", "za", "mzansi", "ZA"],
    ["South Sudan", "republic of south sudan", "SS"],
    ["Spain", "kingdom of spain", "españa", "espana", "ES"],
    ["Sri Lanka", "democratic socialist republic of sri lanka", "ceylon", "LK"],
    ["Sudan (the)", "sudan", "republic of the sudan", "the sudan", "SD"],
    ["Suriname", "republic of suriname", "dutch guiana", "SR"],
    ["Sweden", "kingdom of sweden", "sverige", "SE"],
    ["Switzerland", "swiss confederation", "schweiz", "suisse", "svizzera", "svizra", "CH"],
    ["Syrian Arab Republic (the)", "syria", "syrian arab republic", "the syrian arab republic", "SY"],

    # T
    ["Tajikistan", "republic of tajikistan", "tojikiston", "TJ"],
    ["Tanzania, United Republic of", "tanzania", "united republic of tanzania", "TZ"],
    ["Thailand", "kingdom of thailand", "siam", "TH"],
    ["Timor-Leste", "east timor", "democratic republic of timor-leste", "TL"],
    ["Togo", "togolese republic", "république togolaise", "republique togolaise", "TG"],
    ["Tonga", "kingdom of tonga", "TO"],
    ["Trinidad and Tobago", "trinidad", "tobago", "tt", "TT"],
    ["Tunisia", "tunisian republic", "tunis", "TN"],
    ["Türkiye", "turkiye", "turkey", "republic of türkiye", "republic of turkey", "TR"],
    ["Turkmenistan", "republic of turkmenistan", "TM"],
    ["Tuvalu", "ellice islands", "TV"],

    # U
    ["Uganda", "republic of uganda", "UG"],
    ["Ukraine", "ukraїna", "ukraina", "UA"],
    ["United Arab Emirates (the)", "united arab emirates", "uae", "emirates", "the united arab emirates", "AE"],
    ["United Kingdom of Great Britain and Northern Ireland (the)",
     "united kingdom", "uk", "great britain", "britain", "the united kingdom", "england", "northern ireland", "scotland", "wales",
     "gb", "Bermuda", "somers isles", "Cayman Islands", "cayman", "British Virgin Islands", "bvi", "virgin islands", 
     "Turks and Caicos Islands", "tci", "Anguilla", "Gibraltar", "Montserrat", "Pitcairn Islands", "pitcairn", 
     "pitcairn, henderson, ducie and oeno islands", "Saint Helena, Ascension and Tristan da Cunha", "saint helena", "st helena", 
     "ascension", "tristan da cunha", "British Indian Ocean Territory", "biot", "chagos archipelago", 
     "Falkland Islands (the) [Malvinas]", "falkland islands", "malvinas", "the falkland islands", 
     "GB", "JE", "GG", "IM", "BM", "FK", "TC", "VG", "KY", "MS", "AI"],
    ["United States of America (the)",
     "united states", "usa", "united states of america", "us", "the united states", "america", "US", "u.s.a.", "u.s.", 
     "Puerto Rico", "pr", "PR", 
     "commonwealth of puerto rico", "Guam", "gu", "guåhan", "guahan", "American Samoa", "as", "amerika sāmoa", "amerika samoa", 
     "U.S. Virgin Islands", "us virgin islands", "virgin islands of the united states", "usvi", "VI", 
     "Northern Mariana Islands", "cnmi", "commonwealth of the northern mariana islands"], 
    ["Uruguay", "oriental republic of uruguay", "república oriental del uruguay", "UY"],
    ["Uzbekistan", "republic of uzbekistan", "o'zbekiston", "ozbekiston", "UZ"],

    # V
    ["Vanuatu", "republic of vanuatu", "new hebrides", "VU"],
    ["Venezuela (Bolivarian Republic of)", "venezuela", "bolivarian republic of venezuela", "VE"],
    ["Viet Nam", "vietnam", "socialist republic of vietnam", "vn", "VN"],

    # Y
    ["Yemen", "republic of yemen", "al-yaman", "YE"],

    # Z
    ["Zambia", "republic of zambia", "northern rhodesia", "ZM"],
    ["Zimbabwe", "republic of zimbabwe", "southern rhodesia", "rhodesia", "ZW"],
]

COUNTRY_ALIASES = {}
for group in COUNTRY_GROUPS:
    for country_name in group:
        COUNTRY_ALIASES[country_name.lower()] = group

def standardize_country_name(name: str) -> str:
    """Standardize country name to match the format in the dataset"""
    if not name:
        return ""
    
    normalized = name.strip().lower()
   
    if normalized in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[normalized][0]

    return name.strip()


def are_same_country(country1: str, country2: str) -> bool:
    """
    Check if two country names refer to the same canonical country.
    
    Args:
        country1: First country name to compare
        country2: Second country name to compare
        
    Returns:
        True if both names refer to the same canonical country, False otherwise
    """
    if not country1 or not country2:
        return False
    
    norm1 = country1.strip().lower()
    norm2 = country2.strip().lower()

    if norm1 == norm2:
        return True

    if norm1 in COUNTRY_ALIASES and norm2 in COUNTRY_ALIASES:
        canonical1 = COUNTRY_ALIASES[norm1][0]
        canonical2 = COUNTRY_ALIASES[norm2][0]

        return canonical1 == canonical2

    return False

if __name__ == '__main__':
    pass