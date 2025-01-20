// redux/slices/userSlice.js
import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import {
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signOut,
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

    const user = newUserSnapshot.user;
    const userData = {
      uid: user.uid,
      email: user.email,
      displayName: user.displayName, // if available
      // Add any other fields you need
    };

    const collection = "users";
    await setDoc(doc(db, collection, user.uid), {
      id: user.uid,
      name: payload.name,
      email: payload.email,
      username: payload.username,
      password: payload.password,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    });

    return { success: true, user: userData };
  } catch (err) {
    console.log("error: ", err);
    const errorCode = err.code || "unknown";
    let errorMessage =
      "An unknown error occurred during sign-in. Please try again later.";

    throw new Error({ errorCode, errorMessage }); // Throw as an error object
  }
});

export const login = createAsyncThunk(
  "login",
  async (signInData, { rejectWithValue }) => {
    try {
      const auth = getAuth(firebaseApp);
      const userResponse = await signInWithEmailAndPassword(
        auth,
        signInData.email,
        signInData.password
      );

      const user = userResponse?.user;
      console.log("🚀 ~ user:", user);

      if (user) {
        return {
          success: true,
          user: { uid: user.uid, email: user.email },
        };
      }

      return rejectWithValue({
        message: "User not found in response.",
        code: "200",
      });
    } catch (err) {
      return rejectWithValue(handleAuthError(err)); // Call your error handler
    }
  }
);

export const logout = createAsyncThunk(
  "logout",
  async (_, { rejectWithValue }) => {
    try {
      const auth = getAuth(firebaseApp);
      await signOut(auth);
      return { success: true };
    } catch (err) {
      return rejectWithValue({
        success: false,
        message: err.message || "Failed to sign out.",
      });
    }
  }
);

export const isUserSignedIn = createAsyncThunk("isUserSignedIn", async () => {
  try {
    const auth = getAuth(firebaseApp);
    const user = await new Promise((resolve, reject) => {
      onAuthStateChanged(
        auth,
        (user) => resolve(user || null),
        (error) => reject(error)
      );
    });
    const userDetails = {
      userId: user.uid,
      email: user.email,
      token: user.accessToken,
    };
    return userDetails;
  } catch (error) {
    throw {
      success: false,
      message: error.message || "An error occurred while checking auth state",
    };
  }
});

// Error handling function
const handleAuthError = (error) => {
  // You can extract specific error codes or messages from Firebase Auth error.
  const errorCode = error.code || "unknown";
  let errorMessage = "An unknown error occurred. Please try again later.";

  if (errorCode === "auth/user-not-found") {
    errorMessage = "User not found.";
  }

  return { message: errorMessage, code: errorCode };
};

const userSlice = createSlice({
  name: "users",
  initialState: {
    firebaseUsers: [],
    user: null,
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
        state.firebaseUsers.push(action.payload.user); // Correct data handling
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
        state.user = action.payload.user;
      })
      .addCase(login.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload.message;
      });

    // logout case
    builder
      // Sign-out cases
      .addCase(logout.pending, (state) => {
        state.loading = true;
      })
      .addCase(logout.fulfilled, (state) => {
        state.loading = false;
        state.user = null;
        state.token = null; // Clear the user and token from state
      })
      .addCase(logout.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload.message;
      });

    // User signed in
    builder
      .addCase(isUserSignedIn.pending, (state) => {
        state.loading = true;
      })
      .addCase(isUserSignedIn.fulfilled, (state, action) => {
        state.loading = false;
        state.user = action.payload;
      })
      .addCase(isUserSignedIn.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload?.message || "Error checking auth state";
      });
  },
});

export default userSlice.reducer;
