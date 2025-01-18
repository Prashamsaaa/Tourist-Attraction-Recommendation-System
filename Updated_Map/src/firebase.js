import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyAFLnheOmHcNN-CrqDco0E8oEHx6zjjT-c",
  authDomain: "tourist-recommendation-s-f018b.firebaseapp.com",
  projectId: "tourist-recommendation-s-f018b",
  storageBucket: "tourist-recommendation-s-f018b.firebasestorage.app",
  messagingSenderId: "776421147245",
  appId: "1:776421147245:web:0f1e52a2de8400bb2c3564",
  measurementId: "G-869QWS3XLM",
};

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const db = getFirestore();

export { firebaseApp, db };
