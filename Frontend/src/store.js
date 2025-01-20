import { configureStore } from "@reduxjs/toolkit";
import authReducer from "./slices/AuthStoreSlice"
import preferenceReducer from "./slices/PreferenceSlice"

export const store = configureStore({
  reducer: {
    auth: authReducer,
    preference: preferenceReducer
  },
});

export default store;
