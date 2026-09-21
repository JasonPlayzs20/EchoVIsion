from main import model, IL_rect

results = model(IL_rect)

for e in results:
    print(f"the result is {e}")