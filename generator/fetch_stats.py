"""Fetch contribution calendar + stats from the GitHub GraphQL API -> data/stats.json.
Runs inside the GitHub Action (uses GITHUB_TOKEN). With --sample, writes made-up preview data."""
import json, os, sys, datetime as dt, urllib.request, random

LOGIN = json.load(open("profile.json"))["login"]
os.makedirs("data", exist_ok=True)

def streaks(days):
    cur = best = run = 0
    for d in days:
        run = run + 1 if d["count"] > 0 else 0
        best = max(best, run)
    # current streak: allow today to be empty
    k = len(days) - 1
    if k >= 0 and days[k]["count"] == 0: k -= 1
    while k >= 0 and days[k]["count"] > 0: cur += 1; k -= 1
    return cur, best

if "--sample" in sys.argv:
    rnd = random.Random(4)
    start = dt.date.today() - dt.timedelta(days=364)
    days = []
    for i in range(365):
        d = start + dt.timedelta(days=i)
        heat = 0.15 + 0.85 * (i / 364) ** 3
        c = rnd.choice([0, 0, 0, 1, 2, 3, 5, 8]) if rnd.random() < heat + 0.1 else 0
        days.append({"date": d.isoformat(), "count": c})
    tot = sum(d["count"] for d in days); f = 726 / tot
    for d in days: d["count"] = round(d["count"] * f)
    stats = dict(total=sum(d["count"] for d in days), repos=17, month_commits=520, stars=1)
else:
    token = os.environ["GITHUB_TOKEN"]
    now = dt.datetime.now(dt.timezone.utc)
    som = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    q = """query($login:String!,$from:DateTime!,$to:DateTime!){user(login:$login){
      repositories(ownerAffiliations:OWNER,privacy:PUBLIC,first:100){totalCount nodes{stargazerCount}}
      contributionsCollection{restrictedContributionsCount contributionCalendar{totalContributions weeks{contributionDays{date contributionCount}}}}
      month:contributionsCollection(from:$from,to:$to){totalCommitContributions}}}"""
    body = json.dumps({"query": q, "variables": {"login": LOGIN, "from": som.isoformat(), "to": now.isoformat()}}).encode()
    req = urllib.request.Request("https://api.github.com/graphql", body, {"Authorization": f"bearer {token}"})
    u = json.load(urllib.request.urlopen(req))["data"]["user"]
    cal = u["contributionsCollection"]["contributionCalendar"]
    days = [{"date": d["date"], "count": d["contributionCount"]} for w in cal["weeks"] for d in w["contributionDays"]]
    stats = dict(total=cal["totalContributions"] + u["contributionsCollection"]["restrictedContributionsCount"], repos=u["repositories"]["totalCount"],
                 month_commits=u["month"]["totalCommitContributions"],
                 stars=sum(n["stargazerCount"] for n in u["repositories"]["nodes"]))
cur, best = streaks(days)
stats.update(streak=cur, best_streak=best, month=dt.date.today().strftime("%B").lower(), days=days)
json.dump(stats, open("data/stats.json", "w"))
print({k: v for k, v in stats.items() if k != "days"})
