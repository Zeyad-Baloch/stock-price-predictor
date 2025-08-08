import os
import pickle
import tensorflow as tf
import streamlit as st


class ModelLoader:
    def __init__(self, model_dir="trained_stock_models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.config = None
        self.available_tickers = []

    @st.cache_resource
    def load_config(_self):
        """Load training configuration"""
        try:
            config_path = os.path.join(_self.model_dir, "training_config.pkl")
            with open(config_path, 'rb') as f:
                _self.config = pickle.load(f)
            return _self.config
        except Exception as e:
            st.error(f"Failed to load config: {str(e)}")
            return None

    @st.cache_resource
    def load_ticker_list(_self):
        """Load available ticker symbols"""
        try:
            ticker_path = os.path.join(_self.model_dir, "ticker_list.txt")
            with open(ticker_path, 'r') as f:
                _self.available_tickers = [line.strip() for line in f.readlines()]
            return _self.available_tickers
        except Exception as e:
            st.error(f"Failed to load ticker list: {str(e)}")
            return []

    @st.cache_resource
    def load_model(_self, ticker):
        """Load a specific model and scaler for a ticker"""
        if ticker in _self.models:
            return _self.models[ticker], _self.scalers[ticker]

        try:

            model_path = os.path.join(_self.model_dir, f"{ticker}_model.h5")


            custom_objects = {
                'mse': tf.keras.metrics.MeanSquaredError(),
                'mae': tf.keras.metrics.MeanAbsoluteError(),
                'rmse': tf.keras.metrics.RootMeanSquaredError(),

            }


            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


            scaler_path = os.path.join(_self.model_dir, f"{ticker}_scaler.pkl")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            _self.models[ticker] = model
            _self.scalers[ticker] = scaler

            return model, scaler

        except Exception as e:
            st.error(f"Failed to load model for {ticker}: {str(e)}")
            return None, None

    def get_model_info(self, ticker):
        if not self.config:
            self.load_config()

        if ticker in self.config.get('metrics', {}):
            return self.config['metrics'][ticker]
        return None

    def list_available_models(self):
        """List all available trained models"""
        if not self.available_tickers:
            self.load_ticker_list()
        return self.available_tickers