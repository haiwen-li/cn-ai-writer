library(tidyverse)
library(lmerTest)
library(lme4)
library(clubSandwich)
setwd("~/cn-ai-writer/code")

ratings_df <- read_csv("data/ratings_analysis_df.csv", col_types = 
                         cols(noteId = col_character(), ratedOnTweetId=col_character(), tweetId=col_character()))

ratings_df$AI = as.factor(ratings_df$AI)
ratings_df <- ratings_df %>%
  mutate(
    rater_group = case_when(
      coreRaterFactor1 < -0.15 ~ "left",
      coreRaterFactor1 > 0.15 ~ "right",
      TRUE ~ "middle"
    )
  )
ratings_df$rater_group = factor(ratings_df$rater_group, levels=c("middle", "left", "right"))

# table 1; note + rater re
m1 = lmer(rating_score ~ AI * coreRaterFactor1 + AI * I(coreRaterFactor1^2) + (1|noteId) + (1|raterParticipantId), 
     data=ratings_df,
     control = lmerControl(optimizer = "bobyqa", 
                           optCtrl = list(maxfun = 100000)))

# tweet + rater re
m2 = lmer(rating_score ~AI * coreRaterFactor1+ AI * I(coreRaterFactor1^2)  + (1|tweetId) + (1|raterParticipantId), 
             data=ratings_df,
             control = lmerControl(optimizer = "bobyqa", 
                                   optCtrl = list(maxfun = 100000)))

# ols
m3 = lm(rating_score ~ AI * coreRaterFactor1+ AI * I(coreRaterFactor1^2), data = ratings_df)
vcov_m3 <- vcovCR(m3, cluster = ratings_df$noteId, type = "CR2")
#summary(m3, vcov=vcov_m3)

# by rater group, note + rater re
m4 = lmer(rating_score ~AI * rater_group + (1|noteId) + (1|raterParticipantId), 
          data=ratings_df,
          control = lmerControl(optimizer = "bobyqa", 
                                optCtrl = list(maxfun = 100000)))



###################
library(tidyverse)
library(readr)
library(lme4)
library(broom.mixed)

# --- data ---
tweet_subgroups <- read_csv("data/tweet_subgroups.csv", col_types = 
                              cols(tweetId = col_character()))
df <- left_join(ratings_df, tweet_subgroups, by = "tweetId")


# --- model formula same as m1 ---
form <- rating_score ~ AI * coreRaterFactor1 + AI * I(coreRaterFactor1^2) +
  (1 | noteId) + (1 | raterParticipantId)

ctrl <- lmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 100000)
)

# Helper: robust-ish fit wrapper so the loop doesn't crash on one bad topic
fit_one_subgroup <- function(d) {
  tryCatch(
    lmer(form, data = d, control = ctrl),
    error = function(e) NULL
  )
}

topics <- df %>%
  distinct(topic) %>%
  filter(!is.na(topic)) %>%
  pull(topic)

modality <- df %>%
  distinct(multimodal) %>%
  filter(!is.na(multimodal)) %>%
  pull(multimodal)

results <- map_dfr(topics, function(tpc) {
  d_t <- df %>% filter(topic == tpc)
  
  # You can set a minimum size to avoid unstable fits
  n_obs <- nrow(d_t)
  n_note <- n_distinct(d_t$noteId)
  n_rater <- n_distinct(d_t$raterParticipantId)
  
  mod <- fit_one_subgroup(d_t)
  if (is.null(mod)) {
    return(tibble(
      topic = tpc, n_obs = n_obs, n_note = n_note, n_rater = n_rater,
      term = NA_character_, estimate = NA_real_, std.error = NA_real_
    ))
  }
  
  # Extract fixed effects table
  tt <- broom.mixed::tidy(mod, effects = "fixed") %>%
    mutate(topic = tpc, n_obs = n_obs, n_note = n_note, n_rater = n_rater)
  
  # Identify the AI main effect term name
  # If AI is numeric 0/1 -> "AI"
  # If AI is factor -> usually something like "AIAI" or "AI<level>"
  ai_term <- tt %>%
    filter(term == "AI") %>%
    dplyr::slice(1)
  
  # Fallback for factor-coded AI
  if (nrow(ai_term) == 0) {
    ai_term <- tt %>%
      filter(str_detect(term, "^AI")) %>%
      filter(!str_detect(term, ":")) %>%  # keep main effect, not interactions
      dplyr::slice(1)
  }
  
  if (nrow(ai_term) == 0) {
    return(tibble(
      topic = tpc, n_obs = n_obs, n_note = n_note, n_rater = n_rater,
      term = NA_character_, estimate = NA_real_, std.error = NA_real_
    ))
  }
  
  ai_term %>%
    select(topic, n_obs, n_note, n_rater, term, estimate, std.error)
})

# Compute 95% Wald CI (quick + standard; for more accuracy you can use bootstraps)
results <- results %>%
  mutate(
    conf.low = estimate - 1.96 * std.error,
    conf.high = estimate + 1.96 * std.error
  ) %>%
  filter(!is.na(estimate), !is.na(std.error)) %>%
  arrange(estimate) %>%
  mutate(topic = fct_inorder(topic))

# --- plot: y=topic, x=AI effect, point + CI, dashed red line at 0 ---
p <- ggplot(results, aes(x = estimate, y = topic)) +
  theme_bw(base_size = 12) +
  theme(
    panel.grid.major.y = element_line(color = "grey80", linetype = "dashed", linewidth = 0.5),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_rect(fill = NA, linewidth = 0.8),
    plot.title   = element_text(face = "bold", size = rel(1.05), hjust = 0.5),
    axis.title.y = element_text(margin = margin(r = 8)),
    axis.title.x = element_text(margin = margin(t = 8))
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 0.7) +
  geom_linerange(aes(xmin = conf.low, xmax = conf.high), linewidth = 0.6, color = "grey50") +
  geom_point(size = 2.5, color = "grey30") +
  labs(
    x = "Coefficient of AI (95% CI)",
    y = "Tweet topic",
  )
print(p)
ggsave("outputs/hte_topic.png", p,
       width = 7, height = 4,
       device = png, type = "cairo")


results_m <- map_dfr(modality, function(tpc) {
  d_t <- df %>% filter(multimodal == tpc)
  
  # You can set a minimum size to avoid unstable fits
  n_obs <- nrow(d_t)
  n_note <- n_distinct(d_t$noteId)
  n_rater <- n_distinct(d_t$raterParticipantId)
  
  mod <- fit_one_subgroup(d_t)
  if (is.null(mod)) {
    return(tibble(
      modality = tpc, n_obs = n_obs, n_note = n_note, n_rater = n_rater,
      term = NA_character_, estimate = NA_real_, std.error = NA_real_
    ))
  }
  
  # Extract fixed effects table
  tt <- broom.mixed::tidy(mod, effects = "fixed") %>%
    mutate(modality = tpc, n_obs = n_obs, n_note = n_note, n_rater = n_rater)
  
  # Identify the AI main effect term name
  # If AI is numeric 0/1 -> "AI"
  # If AI is factor -> usually something like "AIAI" or "AI<level>"
  ai_term <- tt %>%
    filter(term == "AI") %>%
    dplyr::slice(1)
  
  # Fallback for factor-coded AI
  if (nrow(ai_term) == 0) {
    ai_term <- tt %>%
      filter(str_detect(term, "^AI")) %>%
      filter(!str_detect(term, ":")) %>%  # keep main effect, not interactions
      dplyr::slice(1)
  }
  
  if (nrow(ai_term) == 0) {
    return(tibble(
      modality = tpc, n_obs = n_obs, n_note = n_note, n_rater = n_rater,
      term = NA_character_, estimate = NA_real_, std.error = NA_real_
    ))
  }
  
  ai_term %>%
    select(modality, n_obs, n_note, n_rater, term, estimate, std.error)
})

# Compute 95% Wald CI (quick + standard; for more accuracy you can use bootstraps)
results_m <- results_m %>%
  mutate(
    conf.low = estimate - 1.96 * std.error,
    conf.high = estimate + 1.96 * std.error
  ) %>%
  filter(!is.na(estimate), !is.na(std.error)) %>%
  arrange(estimate) %>%
  mutate(modality = fct_inorder(modality))


p_m <- ggplot(results_m, aes(x = estimate, y = modality)) +
  # panel border + clean background
  theme_bw(base_size = 12) +
  theme(
    panel.grid.major.y = element_line(color = "grey80", linetype = "dashed", linewidth = 0.5),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.border       = element_rect(fill = NA, linewidth = 0.8),
    plot.title   = element_text(face = "bold", size = rel(1.05), hjust = 0.5),
    axis.title.y = element_text(margin = margin(r = 8)),
    axis.title.x = element_text(margin = margin(t = 8))
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", linewidth = 0.7) +
  geom_linerange(aes(xmin = conf.low, xmax = conf.high), linewidth = 0.6, color = "grey50") +
  geom_point(size = 2.5, color = "grey30") +
  labs(
    x = "Coefficient of AI (95% CI)",
    y = "Tweet modality",
  )
print(p_m)

ggsave("outputs/hte_modality.png", p_m,
       width = 7, height = 4,
       device = png, type = "cairo")

