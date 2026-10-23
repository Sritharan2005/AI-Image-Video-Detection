import urllib.request
import json

def test_wikimedia():
    print("Testing Wikimedia Commons API for authentic workplace photos...")
    url = "https://commons.wikimedia.org/w/api.php?action=query&generator=search&gsrnamespace=6&gsrsearch=office%20worker%20laptop&gsrlimit=5&prop=imageinfo&iiprop=url|size|extmetadata&format=json"
    headers = {"User-Agent": "DeepGuard-Acquisition-Audit/1.0 (https://github.com/deepguard; contact@deepguard.ai)"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode())
        pages = data.get("query", {}).get("pages", {})
        print(f"Wikimedia API returned {len(pages)} valid records.")
        for pid, page in list(pages.items())[:2]:
            ii = page.get("imageinfo", [{}])[0]
            meta = ii.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value")
            artist = meta.get("Artist", {}).get("value")
            print(f"  Title: {page.get('title')}")
            print(f"  URL: {ii.get('url')}")
            print(f"  License: {lic}, Author: {str(artist)[:40] if artist else 'Unknown'}")
            print(f"  Size: {ii.get('width')}x{ii.get('height')}")
        return True
    except Exception as e:
        print(f"Wikimedia API test failed: {e}")
        return False

def test_huggingface_openfake():
    print("\nTesting Hugging Face OpenFake Server API for AI generation provenance...")
    url = "https://datasets-server.huggingface.co/rows?dataset=ComplexDataLab/OpenFake&config=core&split=train&offset=0&limit=5"
    headers = {"User-Agent": "DeepGuard-Acquisition-Audit/1.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        rows = data.get("rows", [])
        print(f"Hugging Face API returned {len(rows)} records.")
        for r in rows[:2]:
            row = r["row"]
            print(f"  Row ID: {row.get('row_id')}, Label: {row.get('label')}, Model: {row.get('model')}")
            print(f"  Image URL: {row.get('image', {}).get('src')[:60]}...")
        return True
    except Exception as e:
        print(f"Hugging Face API test failed: {e}")
        return False

if __name__ == "__main__":
    test_wikimedia()
    test_huggingface_openfake()
