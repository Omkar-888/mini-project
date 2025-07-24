package main

import (
	"net/http"
	"os"

	"github.com/joho/godotenv"
	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
	"github.com/rs/zerolog/log"
)

func GetIndexHandler(c echo.Context) error {
	return c.JSON(http.StatusOK, map[string]any{
		"message": "Welcome to this Mini Project",
		"routes": map[string]string{
			"/health": "health check",
		},
	})
}

func GetHealthHandler(c echo.Context) error {
	return c.JSON(http.StatusOK, map[string]string{
		"status": "ok",
	})
}

func main() {
	err := godotenv.Load(".env")
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load .env file.")
	}

	port := os.Getenv("PORT")
	if port == "" {
		log.Fatal().Msg("PORT variable not found in .env.")
	}
	port = ":" + port

	r := echo.New()
	r.Use(middleware.Logger())
	r.Use(middleware.Recover())
	r.GET("/", GetIndexHandler)
	r.GET("/health", GetHealthHandler)

	err = r.Start(port)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to start server.")
	}
}
