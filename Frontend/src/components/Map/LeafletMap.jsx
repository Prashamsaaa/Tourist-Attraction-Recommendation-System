import { MapContainer, Marker, Popup, TileLayer } from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility/dist/leaflet-defaulticon-compatibility.css";
import "leaflet-defaulticon-compatibility";

const LeafletMap = () => {
  const place = {
    name: "Eiffel Tower",
    location: [48.8584, 2.2945],
    address: "Champ de Mars, 5 Avenue Anatole France, 75007 Paris, France",
    description:
      "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
    itinerary: [
      "9:00 AM - Arrive at the Eiffel Tower",
      "9:30 AM - Take the elevator to the second floor",
      "10:30 AM - Visit the top floor for panoramic views",
      "11:30 AM - Enjoy lunch at the 58 Tour Eiffel restaurant",
      "1:00 PM - Explore the Champ de Mars park",
    ],
    website: "https://www.toureiffel.paris/en",
  };
  return (
    <div>
      <MapContainer
        center={place.location}
        zoom={15}
        scrollWheelZoom={false}
        style={{ height: "100%", width: "100%" }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Marker position={place.location}>
          <Popup>{place.name}</Popup>
        </Marker>
      </MapContainer>
    </div>
  );
};

export default LeafletMap;
