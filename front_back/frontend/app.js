const STORAGE_KEY = "empathia-chats-v1";
const THEME_KEY = "empathia-theme";

const state = {
  chats: loadChats(),
  activeId: null,
  sending: false,
};

const elements = {
  conversation: document.querySelector("#conversation"),
  welcome: document.querySelector("#welcomeScreen"),
  messages: document.querySelector("#messages"),
  input: document.querySelector("#messageInput"),
  form: document.querySelector("#chatForm"),
  send: document.querySelector("#sendButton"),
  history: document.querySelector("#chatHistory"),
  newChat: document.querySelector("#newChatButton"),
  clearHistory: document.querySelector("#clearHistoryButton"),
  sidebar: document.querySelector("#sidebar"),
  overlay: document.querySelector("#mobileOverlay"),
  openSidebar: document.querySelector("#openSidebar"),
  closeSidebar: document.querySelector("#closeSidebar"),
  theme: document.querySelector("#themeButton"),
  toast: document.querySelector("#toast"),
};

init();

function init() {
  const savedTheme = localStorage.getItem(THEME_KEY);
  if (savedTheme === "dark") document.documentElement.dataset.theme = "dark";

  bindEvents();
  renderHistory();
  autoResize();
}

function bindEvents() {
  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(elements.input.value);
  });

  elements.input.addEventListener("input", () => {
    autoResize();
    updateSendButton();
  });

  elements.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage(elements.input.value);
    }
  });

  document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => sendMessage(button.dataset.prompt));
  });

  elements.newChat.addEventListener("click", newChat);
  elements.clearHistory.addEventListener("click", clearHistory);
  elements.openSidebar.addEventListener("click", openSidebar);
  elements.closeSidebar.addEventListener("click", closeSidebar);
  elements.overlay.addEventListener("click", closeSidebar);
  elements.theme.addEventListener("click", toggleTheme);

  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      newChat();
    }
  });
}

async function sendMessage(rawMessage) {
  const message = rawMessage.trim();
  if (!message || state.sending) return;

  let chat = getActiveChat();
  if (!chat) {
    chat = {
      id: crypto.randomUUID(),
      title: makeTitle(message),
      createdAt: Date.now(),
      messages: [],
    };
    state.chats.unshift(chat);
    state.activeId = chat.id;
  }

  chat.messages.push({ role: "user", content: message });
  elements.input.value = "";
  state.sending = true;
  autoResize();
  updateSendButton();
  saveChats();
  renderHistory();
  renderConversation(true);
  closeSidebar();

  const startedAt = Date.now();

  try {
    const response = await fetch("/api/chat/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: chat.id, message }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    const remainingDelay = Math.max(0, 650 - (Date.now() - startedAt));
    await wait(remainingDelay);

    chat.messages.push({
      role: "assistant",
      content: payload.message,
      crisis: payload.crisis_detected,
    });
  } catch (error) {
    chat.messages.push({
      role: "assistant",
      content: "Не получилось связаться с сервером. Проверь, что FastAPI запущен, и попробуй ещё раз.",
      error: true,
    });
    showToast("Сервер недоступен");
  } finally {
    state.sending = false;
    saveChats();
    renderConversation();
    updateSendButton();
    elements.input.focus();
  }
}

function renderConversation(showTyping = false) {
  const chat = getActiveChat();
  const hasMessages = chat?.messages.length > 0;
  elements.welcome.hidden = Boolean(hasMessages);
  elements.messages.classList.toggle("active", Boolean(hasMessages));

  if (!hasMessages) {
    elements.messages.innerHTML = "";
    return;
  }

  elements.messages.innerHTML = chat.messages.map(messageTemplate).join("");
  if (showTyping) elements.messages.insertAdjacentHTML("beforeend", typingTemplate());
  requestAnimationFrame(scrollToBottom);
}

function messageTemplate(message) {
  if (message.role === "user") {
    return `
      <article class="message-row user">
        <div class="message-content"><div class="message-text">${escapeHtml(message.content)}</div></div>
      </article>`;
  }

  const extraClass = message.crisis ? " crisis" : "";
  return `
    <article class="message-row assistant${extraClass}">
      <div class="message-avatar">э</div>
      <div class="message-content">
        <div class="message-name">Эмпатия</div>
        <div class="message-text">${escapeHtml(message.content)}</div>
      </div>
    </article>`;
}

function typingTemplate() {
  return `
    <article class="message-row assistant" id="typingMessage">
      <div class="message-avatar">э</div>
      <div class="message-content">
        <div class="message-name">Эмпатия размышляет</div>
        <div class="typing"><i></i><i></i><i></i></div>
      </div>
    </article>`;
}

function renderHistory() {
  if (!state.chats.length) {
    elements.history.innerHTML = `<div class="history-item" style="cursor:default;opacity:.65"><span>Здесь появятся разговоры</span></div>`;
    return;
  }

  elements.history.innerHTML = state.chats.map((chat) => `
    <button class="history-item ${chat.id === state.activeId ? "active" : ""}" data-chat-id="${chat.id}">
      <svg viewBox="0 0 24 24"><path d="M21 12a8 8 0 0 1-9 8 9 9 0 0 1-4-.9L3 21l1.7-4.5A8 8 0 1 1 21 12Z"/></svg>
      <span>${escapeHtml(chat.title)}</span>
    </button>`).join("");

  elements.history.querySelectorAll("[data-chat-id]").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeId = button.dataset.chatId;
      renderHistory();
      renderConversation();
      closeSidebar();
    });
  });
}

function newChat() {
  state.activeId = null;
  state.sending = false;
  elements.input.value = "";
  renderHistory();
  renderConversation();
  updateSendButton();
  closeSidebar();
  elements.input.focus();
}

async function clearHistory() {
  if (!state.chats.length) return showToast("История уже пуста");
  if (!window.confirm("Удалить все разговоры из этого браузера?")) return;

  const sessionIds = state.chats.map((chat) => chat.id);
  state.chats = [];
  state.activeId = null;
  saveChats();
  renderHistory();
  renderConversation();
  showToast("История удалена");

  await Promise.allSettled(sessionIds.map((sessionId) => fetch("/api/chat/session", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  })));
}

function toggleTheme() {
  const isDark = document.documentElement.dataset.theme === "dark";
  if (isDark) delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = "dark";
  localStorage.setItem(THEME_KEY, isDark ? "light" : "dark");
}

function getActiveChat() {
  return state.chats.find((chat) => chat.id === state.activeId);
}

function loadChats() {
  try {
    const chats = JSON.parse(localStorage.getItem(STORAGE_KEY));
    return Array.isArray(chats) ? chats : [];
  } catch {
    return [];
  }
}

function saveChats() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.chats));
}

function makeTitle(message) {
  const clean = message.replace(/\s+/g, " ");
  return clean.length > 34 ? `${clean.slice(0, 34)}…` : clean;
}

function autoResize() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`;
}

function updateSendButton() {
  elements.send.disabled = state.sending || !elements.input.value.trim();
  elements.input.disabled = state.sending;
}

function scrollToBottom() {
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
}

function openSidebar() {
  elements.sidebar.classList.add("open");
  elements.overlay.classList.add("open");
}

function closeSidebar() {
  elements.sidebar.classList.remove("open");
  elements.overlay.classList.remove("open");
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("show");
  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => elements.toast.classList.remove("show"), 2200);
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
