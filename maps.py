import requests


def get_nearby_recycling_centers(lat, lng):
    api_key = "AIzaSyDynUDk-JACKfSBMhKIrGXjU1q2pjqqrRY"
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=5000&type=recycling_center&key={api_key}"

    response = requests.get(url)
    data = response.json()

    nearby_centers = []
    for result in data["results"]:
        nearby_centers.append(
            {"name": result["name"], "address": result.get("vicinity", "Not available")}
        )

    return nearby_centers
