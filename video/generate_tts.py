"""Generate TTS narration for all 8 scenes using edge-tts."""
import asyncio
import edge_tts
import os

ASSETS = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSETS, exist_ok=True)

VOICE = "zh-CN-YunjianNeural"  # Male, Passionate, Sports/Novel style
CLOSING_VOICE = "zh-CN-YunyangNeural"  # Male, Professional, Authoritative
RATE = "+35%"  # Faster pace - not too slow

scenes = [
    (1, VOICE,
     "五十家新創公司。一個核心問題：什麼因素才是真正驅動企業利潤的關鍵？五位領域專家，五輪深入討論，三位機器學習模型，一個不容置疑的答案。"
    ),
    (2, VOICE,
     "研發投資。行銷推廣。行政管理。地區選擇。五個關鍵變數，哪一個才是利潤的核心引擎？專家們各持己見，數據將給出最終裁決。"
    ),
    (3, VOICE,
     "探索性資料分析，揭示真相。研發支出與利潤的相關係數高達零點九七——壓倒性的證據！行銷支出以零點七五次之。行政管理的貢獻微乎其微，地區因素的影響幾乎可以忽略不計。"
    ),
    (4, VOICE,
     "三個機器學習模型，交叉驗證。線性回歸。隨機森林。XGBoost。隨機森林以R平方零點九一奪冠。三個模型的結論高度一致：研發支出，是無可爭議的核心驅動變數。"
    ),
    (5, VOICE,
     "特徵重要性排名震撼出爐。研發支出，獨占百分之九十二點零的重要性！行銷支出止於百分之四點八。行政管理，百分之一點九。地區因素，不到百分之一點三。差距之懸殊，超乎想像。"
    ),
    (6, VOICE,
     "五位專家，全數投票。研發總監信心高達九十二分。行銷總監承認研發的主導地位。行政總監接受數據的事實。州長同意資源的優先順序。機器學習專家確認證據已充分。研發為王，數據為證。"
    ),
    (7, VOICE,
     "終極預算配置。研發投入，百分之六十。最強預測貢獻。行銷投入，百分之二十五。與研發相輔相成。行政管理，百分之十。營運基礎支撐。保留備用，百分之五。這是數據驅動的最優解。"
    ),
    (8, CLOSING_VOICE,
     "研發為王。數據不騙人。以科學的態度面對決策，用數據的語言回答問題。五十家新創企業的故事給出一個答案：投資研發，就是投資未來。"
    ),
]

async def gen(scene_num, voice, text):
    out = os.path.join(ASSETS, f"scene{scene_num}_narration.mp3")
    communicate = edge_tts.Communicate(text, voice, rate=RATE, pitch="-2Hz")
    await communicate.save(out)
    print(f"  Scene {scene_num}: {len(text)} chars -> {out}")

async def main():
    print(f"Voice: {VOICE}, Rate: {RATE}")
    for s in scenes:
        await gen(*s)
    print(f"\nDone. {len(scenes)} files saved to {ASSETS}")

asyncio.run(main())
