import { useEffect, useState } from "react";

type Face = {
  id: string;
  age: number;
  gender: string;
  emotion: string;
  img_data_url: string;  // base64 jpeg
};

export default function FaceCrops() {
  const [faces, setFaces] = useState<Face[]>([]);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8082/faces"); // adjust to your backend
        const payload = await res.json();
        if (payload && payload.faces) {
          setFaces(payload.faces);
        }
      } catch (err) {
        console.error("Failed to fetch faces:", err);
      }
    }, 500); // poll every 500ms

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {faces.map((face, idx) => (
        <div key={idx} style={{
          border: "1px solid #ccc",
          borderRadius: "8px",
          padding: "8px",
          background: "white",
          display: "flex",
          alignItems: "center",
          gap: "12px"
        }}>
          <img
            src={face.img_data_url}
            alt={`Face ${idx}`}
            style={{ width: "120px", height: "120px", objectFit: "cover", borderRadius: "6px" }}
          />
          <div>
            <div><b>ID:</b> {face.id}</div>
            <div><b>Age:</b> {face.age}</div>
            <div><b>Gender:</b> {face.gender}</div>
            <div><b>Emotion:</b> {face.emotion}</div>
          </div>
        </div>
      ))}
    </div>
  );
}
