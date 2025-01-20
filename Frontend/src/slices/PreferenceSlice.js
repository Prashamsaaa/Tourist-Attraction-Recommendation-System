import { createAsyncThunk, createSlice } from "@reduxjs/toolkit";
import { isUserSignedIn } from "./AuthStoreSlice";
import {
  collection,
  doc,
  getDocs,
  query,
  setDoc,
  where,
} from "firebase/firestore";
import { db } from "../firebase";

const collectionName = "preferences";
const collectionRef = collection(db, collectionName);
export const createPreference = createAsyncThunk(
  "preferences/create",
  async (payload, { dispatch, rejectWithValue }) => {
    try {
      const user = await dispatch(isUserSignedIn()).unwrap();
      if (!user) {
        throw new Error("User not found");
      }

      await setDoc(
        doc(db, collectionName, user.userId),
        {
          id: user.userId,
          preference: payload,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        },
        { merge: true }
      );

      console.log("Preference saved successfully.");
      return { success: true, message: "Preference saved successfully." };
    } catch (error) {
      console.error("Error saving preference:", error);

      return rejectWithValue({
        success: false,
        message:
          error.code === "permission-denied"
            ? "You do not have permission to add a preference."
            : error.message || "Unknown error",
        code: error.code || "unknown",
      });
    }
  }
);

export const getPreferenceByUserId = createAsyncThunk(
  "preferences/get",
  async (_, { dispatch }) => {
    try {
      const user = await dispatch(isUserSignedIn()).unwrap();
      if (!user) {
        throw new Error("User not found");
      }
      const userId = user.userId;

      const preferenceQuery = query(collectionRef, where("id", "==", userId));
      const querySnapshot = await getDocs(preferenceQuery);

      if (querySnapshot.empty) {
        console.log("No preference found for user");

        return null;
      }

      // Extract first document
      const preferenceDoc = querySnapshot.docs[0].data();

      return preferenceDoc;
    } catch (error) {
      console.log("Error in fetching preference of user", error);
    }
  }
);

const preferenceSlice = createSlice({
  name: "preferences",
  initialState: {
    loading: false,
    fetching: false,
    error: null,
    userPreferenceData: {},
  },
  reducers: {},
  extraReducers: (builder) => {
    // Creating Preference of User
    builder
      .addCase(createPreference.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(createPreference.fulfilled, (state) => {
        state.loading = false;
      })
      .addCase(createPreference.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.message || "An error occurred.";
      });

    // Getting preference list
    builder
      .addCase(getPreferenceByUserId.pending, (state) => {
        state.fetching = true;
        state.userPreferenceData = {};
        state.error = null;
      })
      .addCase(getPreferenceByUserId.fulfilled, (state, action) => {
        state.fetching = false;
        state.userPreferenceData = action.payload || {};
      })
      .addCase(getPreferenceByUserId.rejected, (state, action) => {
        state.fetching = false;
        state.userPreferenceData = {};
        state.error = action.payload?.message || "An error occurred.";
      });
  },
});

export default preferenceSlice.reducer;
