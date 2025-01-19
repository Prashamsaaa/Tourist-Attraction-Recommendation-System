// redux/slices/userSlice.js
import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
} from "firebase/auth";
import { getAuth } from "firebase/auth";
import { firebaseApp } from "../firebase";
import { doc, setDoc } from "firebase/firestore";
import { db } from "../firebase";

export const createUser = createAsyncThunk("createUser", async (payload) => {
  try {
    const auth = getAuth(firebaseApp);
    const newUserSnapshot = await createUserWithEmailAndPassword(
      auth,
      payload.email,
      payload.password
    );
    const collection = "users";
    await setDoc(doc(db, collection, newUserSnapshot.user.uid), {
      id: newUserSnapshot.user.uid,
      name: payload.name,
      email: payload.email,
      username: payload.username,
      password: payload.password,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    });
    //   await signOut(auth);
    return { success: true, newUserSnapshot };
  } catch (err) {
    console.log("error: ", err);
    const errorCode = err.code || "unknown";
    let errorMessage =
      "An unknown error occurred during sign-in. Please try again later.";

    throw {
      message: errorMessage,
      code: errorCode,
    };
  }
});

export const login = createAsyncThunk("login", async ({email, password}) => {
  try {
    console.log("in login store: ", email, password)
    const auth = getAuth(firebaseApp);
    const signInResponse = await signInWithEmailAndPassword(
      auth,
      email,
      password
    );
    return { success: true, uid: signInResponse.user.uid };
  } catch (err) {
    const errorCode = err.code || "unknown";
    let errorMessage =
      "An unknown error occurred during sign-in. Please try again later.";

    throw {
      message: errorMessage,
      code: errorCode,
    };
  }
});

const userSlice = createSlice({
  name: "users",
  initialState: {
    firebaseUsers: [],
    loading: false,
    error: null,
  },
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(createUser.pending, (state) => {
        state.loading = true;
      })
      .addCase(createUser.fulfilled, (state, action) => {
        state.loading = false;
        state.firebaseUsers = action.payload;
      })
      .addCase(createUser.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });

    // login case
    builder
      .addCase(login.pending, (state) => {
        state.loading = true;
      })
      .addCase(login.fulfilled, (state, action) => {
        state.loading = false;
      })
      .addCase(login.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message;
      });
  },
});

export default userSlice.reducer;
