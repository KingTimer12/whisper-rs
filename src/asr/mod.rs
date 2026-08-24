#[cfg(test)]
mod tests {
    /// The Arc<Ct2Asr> design in the Python layer requires this.
    /// If this fails to compile, Ct2Asr must wrap Whisper in a Mutex.
    #[test]
    fn whisper_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ct2rs::Whisper>();
    }
}
