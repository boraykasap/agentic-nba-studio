# Agentic NBA Studio

Hackathon entry for the **Agentic Next Best Action (NBA) Studio** challenge: put a virtually unlimited number of personalized NBAs at the fingertips of central sales teams and Client Advisors, generated on the fly through natural language.

Built for the [Finnova Hackathon](https://ai-weeks.ch/events/finnova-hackathon) by team **NextBext**: Ozan Erdal, Kerem Işık and Boray Kasap.

## Why it matters

NBA development today is slow, static and one-off. Analytics teams build each NBA for a single campaign, using only part of the available client context. As a result:

- only a fraction of the addressable client base is ever reached, often at the wrong time
- NBAs are hard to adapt when client behaviour or campaign results change
- every new use case starts from scratch

Letting sales teams and advisors create and refine NBAs themselves, in plain language and on top of the full client context, turns intent into conversion without waiting on a data science backlog.

## What it does

| Challenge goal | In the app |
|----------------|-----------|
| **NBA Library** – discover existing NBAs | **Campaigns → Existing Campaigns**: pick an NBA, see its whole target audience, filter by priority and match score |
| **NBA Generation** – create NBAs on the fly | **Campaigns → New Campaign**: describe a product in plain language. The LLM proposes hard eligibility filters plus soft psychographic weights, scored against customer embeddings. Refine by chatting |
| **NBA Editing** – adapt through natural language | Refining a campaign replaces its filters and weights with the updated ones |
| **Explain** – make every NBA transparent | Each NBA shows its confidence and rationale. New campaigns show the exact filters and trait weights used |
| **Client Advisor view** | **Customer**: profile, top-3 NBAs, raw accounts/balances/events/transactions, and a chat grounded in that client's data. The live top recommendation updates as the conversation surfaces new information |

**Not built yet:** predictive NBAs (on-the-fly ML models), and the feedback/learning loop (silencing NBAs, tracking campaign outcomes).

## Run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # needed for chat and new-campaign targeting
streamlit run app.py
```
