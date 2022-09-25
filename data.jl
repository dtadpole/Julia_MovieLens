using CSV
using Nettle
using ZipFile
using DataFrames

FILENAME = "./data/ml-1m.zip"

USERS = nothing
MOVIES = nothing
RATINGS = nothing

function download_file()
    global FILENAME
    if !isfile(FILENAME)
        download("http://files.grouplens.org/datasets/movielens/ml-1m.zip", FILENAME)
        @info "Downloaded data to [$(FILENAME)]"
    end
end

function load_ratings()
    global FILENAME, USERS, MOVIES, RATINGS
    if RATINGS === nothing
        ratingReader = ZipFile.Reader(FILENAME)
        rating_file = filter(x -> x.name == "ml-1m/ratings.dat", ratingReader.files)[1]
        RATINGS = CSV.File(rating_file, delim="::", header=[:uid, :mid, :rating, :timestamp]) |> DataFrame
        RATINGS = sort(RATINGS, [:uid, :timestamp])
        close(ratingReader)
    end
    return RATINGS
end

function load_users()
    global FILENAME, USERS, MOVIES, RATINGS
    if USERS === nothing
        userReader = ZipFile.Reader(FILENAME)
        user_file = filter(x -> x.name == "ml-1m/users.dat", userReader.files)[1]
        USERS = CSV.File(user_file, delim="::", header=[:uid, :gender, :age, :occupation, :other]) |> DataFrame
        close(userReader)
    end
    return USERS
end

function load_movies()
    global FILENAME, USERS, MOVIES, RATINGS
    if MOVIES === nothing
        movieReader = ZipFile.Reader(FILENAME)
        movie_file = filter(x -> x.name == "ml-1m/movies.dat", movieReader.files)[1]
        MOVIES = CSV.File(movie_file, delim="::", header=[:mid, :title, :genres]) |> DataFrame
        close(movieReader)
    end
    return MOVIES
end

if abspath(PROGRAM_FILE) == @__FILE__

    # Print the first 5 rows of the users DataFrame
    @show first(USERS, 5)

    # Print the first 5 rows of the movies DataFrame
    @show first(MOVIES, 5)

    # Print the first 5 rows of the ratings DataFrame
    @show first(RATINGS, 5)

end
