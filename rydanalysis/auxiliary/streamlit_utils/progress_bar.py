import streamlit as st


class streamlit_tqdm:
    def __init__(self, iterable, desc=None, leave=True):
        if desc:
            self.message = st.markdown(desc)
        self.prog_bar = st.progress(0)
        self.iterable = iterable
        self.length = len(iterable)
        self.i = 0
        self.leave = leave

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.i += 1
            current_prog = self.i / self.length
            self.prog_bar.progress(current_prog)
        if self.leave:
            self.prog_bar.empty()
            self.message.empty()
